import pandas as pd
import requests
from tqdm import tqdm
import time, json
import psutil
import os
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stopwords_pt = set(stopwords.words('portuguese'))

process = psutil.Process(os.getpid())
start_time = time.time()

df = pd.read_csv("2_comentarios_pre_processados.csv")
print("Quantidade de comentários:", len(df))

model_name = 'llama3.2'
arquivo_geral = "final_llama32_comentarios_aspectos.csv"
arquivo_ofensivos = "final_llama32_comentarios_ofensivos.csv"
arquivo_sem_aspectos = "final_llama32_comentarios_sem_aspectos.csv"

agrupamentos = {
    r"(preço|custo|custos).*(alto|elevado|caro|altos|excessivo)": "custo alto",
    r"(demora|lento|devagar)": "carregamento lento",
    r"(baixa|limitada|pouca)": "limitada",
    r"(sem carregador|poucos postos|infraestrutura ruim|sem infraestrutura)": "falta de infraestrutura",
    r"(baixa potência|motor.*fraco|subida.*fraca)": "potência baixa",
    r"(revenda ruim|baixa revenda|perda.*valor)": "baixa revenda",
    r"(falta informações sobre veículo elétrico|falta conhecimento sobre tecnologia|falta conhecimento sobre tecnologia veículos elétricos|falta conhecimento sobre carros elétricos|falta conhecimento técnico)\b":"falta de conhecimento técnico",
    r"\b(nenhum aspectos técnicos claros extrair|nenhuma|não há|nao ha|nao tem|não tem|nada|não especificado|nenhum aspecto|none)\b": "nenhum"
}


def normalizar_aspectos(texto):
    if not texto or texto.strip() == "":
        return "nenhum"

    texto = texto.strip().lower()

    texto = re.sub(r"\baspecto\s*\d+\s*[\-,.:;]*\s*", "", texto)

    texto = re.sub(r"[^a-zA-Z0-9áéíóúâêîôûãõç ,]", " ", texto)

    texto = re.sub(r"\s+", " ", texto)

    # Remove duplicatas e stopwords
    partes = [p.strip() for p in texto.split(",") if p.strip()]
    partes_unicas = list(dict.fromkeys(partes))  

    aspectos_filtrados = []
    for aspecto in partes_unicas:
        palavras = [p for p in aspecto.split() if p not in stopwords_pt]
        aspecto_limpo = " ".join(palavras).strip()
        if aspecto_limpo:
            aspectos_filtrados.append(aspecto_limpo)

    if not aspectos_filtrados:
        return "nenhum"

    texto = ", ".join(aspectos_filtrados)

    for padrao, substituto in agrupamentos.items():
        texto = re.sub(padrao, substituto, texto)

    texto = texto.strip(",. ").strip()
    if texto == "" or texto == "aspecto":
        return "nenhum"

    return texto



def extrair_aspectos_ollama(frase):
    prompt = f"""
Você é um especialista técnico em carros elétricos. 
Seu papel é analisar comentários e identificar aspectos técnicos **específicos** sobre **funcionamento**, **tecnologia** ou **características reais** dos veículos elétricos.

Responda no formato (em uma linha, sem aspas):

positivos: aspecto1, aspecto2
negativos: aspecto3, aspecto4

Responda apenas com NENHUM em qualquer outro caso de falta de conhecimento, ou se não possuir informações específicas ou aspectos técnicos, ou linguagem ofensiva .

Frase: "{frase}"
"""
    try:
        url = 'http://localhost:11434/api/generate'
        headers = {
            'Content-Type': 'applicaton/json'
        }
        data={'model': model_name, 'prompt': prompt, 'stream': False}
        response = requests.post(
                url, headers=headers, data=json.dumps(data)
        )
        if response.status_code == 200:
            return response.json()['response'].strip().lower()
        else:
            print("❌ Erro:", response.status_code, response.text)
    except Exception as e:
        print("❌ Exceção:", str(e))
    return "positivos: nenhum, negativos: nenhum"

# Cabeçalhos
escreve_cabecalho_geral = not os.path.exists(arquivo_geral)
escreve_cabecalho_ofensivos = not os.path.exists(arquivo_ofensivos)
escreve_cabecalho_sem_aspectos = not os.path.exists(arquivo_sem_aspectos)

comentarios = df["content"].dropna().drop_duplicates().tolist()

# Regex
regex_both = re.compile(r'positivos?:\s*(.*?)(?:\s+)?negativos?:\s*(.*)', re.IGNORECASE)
regex_pos = re.compile(r'positivos?:\s*(.*)', re.IGNORECASE)
regex_neg = re.compile(r'negativos?:\s*(.*)', re.IGNORECASE)

# Loop
for frase in tqdm(comentarios):
    resposta = extrair_aspectos_ollama(frase)
    print(resposta)

    if any(term in resposta for term in ["linguagem ofensiva", "racismo", "xenofobia", "misoginia", "ameaça", "ameaças"]):
        linha_df = pd.DataFrame([{
            "comentario": frase,
            "resposta": "linguagem ofensiva",
        }])
        linha_df.to_csv(arquivo_ofensivos, mode='a', header=escreve_cabecalho_ofensivos, index=False, encoding='utf-8')
        escreve_cabecalho_ofensivos = False
        continue

    positivos = "nenhum"
    negativos = "nenhum"

    match_both = regex_both.search(resposta)
    if match_both:
        positivos = normalizar_aspectos(match_both.group(1))
        negativos = normalizar_aspectos(match_both.group(2))
    else:
        match_pos = regex_pos.search(resposta)
        if match_pos:
            positivos = normalizar_aspectos(match_pos.group(1))
        match_neg = regex_neg.search(resposta)
        if match_neg:
            negativos = normalizar_aspectos(match_neg.group(1))
    print("\n\nPositivo corrigido: " ,positivos)
    print("\n\nNegativo corrigigo: " ,negativos)
    linha_df = pd.DataFrame([{
        "comentario": frase,
        "positivo": positivos,
        "negativo": negativos
    }])

    if positivos == "nenhum" and negativos == "nenhum":
        linha_df.to_csv(arquivo_sem_aspectos, mode='a', header=escreve_cabecalho_sem_aspectos, index=False, encoding='utf-8')
        escreve_cabecalho_sem_aspectos = False
    else:
        linha_df.to_csv(arquivo_geral, mode='a', header=escreve_cabecalho_geral, index=False, encoding='utf-8')
        escreve_cabecalho_geral = False

# Estatísticas
end_time = time.time()
print("\n✅ Finalizado!")
print(f"⏱ Tempo: {end_time - start_time:.2f}s")
print(f"🧠 Memória usada: {process.memory_info().rss / 1024 ** 2:.2f} MB")
