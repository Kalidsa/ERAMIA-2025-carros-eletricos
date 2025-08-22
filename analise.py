import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import os

df = pd.read_csv("final_llama32_comentarios_aspectos.csv", encoding="utf-8-sig")

substituicoes = {
    "custo altos": "custo alto",
    "alto custo": "custo alto",
    "limitações autonomia" : "autonomia limitada",
    "limitada autonomia" : "autonomia limitada",
    "autonomia longa" : "autonomia prolongada",
    "carregamento carregamento lento" : "carregamento lento",
    "falta infraestrutura carga" : "falta infraestrutura carregamento",
    "falta infraestrutura recarga" : "falta infraestrutura carregamento",
     "limitações carga" : "carga limitada",
    "capacidade carga limitada" : "carga limitada",
}

colunas_para_substituir = ["positivo", "negativo"]

for coluna in colunas_para_substituir:
    df[coluna] = df[coluna].astype(str)  
    for alvo, substituto in substituicoes.items():
        df[coluna] = df[coluna].str.replace(alvo, substituto, regex=False)

df.to_csv("final_llama32_comentarios_aspectos_corrigido.csv", index=False, encoding="utf-8-sig")
df_final = pd.read_csv("final_llama32_comentarios_aspectos_corrigido.csv", encoding="utf-8-sig")

df_final = pd.read_csv("final_llama32_comentarios_aspectos_corrigido.csv", encoding="utf-8-sig")
df_nenhum = pd.read_csv("final_llama32_comentarios_sem_aspectos.csv", encoding="utf-8-sig")
df_ofensivos = pd.read_csv("final_llama32_comentarios_ofensivos.csv", encoding="utf-8-sig")

# Normalize column names
df_final.columns = df_final.columns.str.strip().str.lower()

def process_aspectos(series):
    all_aspects = []
    for entry in series:
        if isinstance(entry, str) and entry.lower().strip() != "nenhum":
            aspectos = [a.strip().lower() for a in entry.split(",") if a.strip()]
            all_aspects.extend(aspectos)
    return all_aspects

positivos = process_aspectos(df_final["positivo"])
negativos = process_aspectos(df_final["negativo"])

positivos_count = Counter(positivos)
negativos_count = Counter(negativos)

top_positivos = pd.DataFrame(positivos_count.most_common(20), columns=["aspect", "frequency"])
top_negativos = pd.DataFrame(negativos_count.most_common(20), columns=["aspect", "frequency"])

# Create output folder
os.makedirs("graficos_llama", exist_ok=True)

# =====================
# 3. Bar charts
# =====================

# =====================
# 3. Bar charts (com valores nas barras)
# =====================

plt.figure(figsize=(14, 6))


plt.subplot(1, 2, 1)
ax1 = sns.barplot(data=top_positivos, y="aspect", x="frequency", color="green")
plt.title("Top 20 Positive Aspects")
for p in ax1.patches:
    width = p.get_width()
    plt.text(
        width + 0.5,                   
        p.get_y() + p.get_height() / 2, 
        int(width),                    
        va="center"
    )

plt.subplot(1, 2, 2)
ax2 = sns.barplot(data=top_negativos, y="aspect", x="frequency", color="red")
plt.title("Top 20 Negative Aspects")
for p in ax2.patches:
    width = p.get_width()
    plt.text(
        width + 0.5,
        p.get_y() + p.get_height() / 2,
        int(width),
        va="center"
    )

plt.tight_layout()
plt.savefig("graficos_llama/top_aspects_bars.png", dpi=300)


# =====================
# 4. Word clouds
# =====================

# Gerar as wordclouds
def green_color_func(*args, **kwargs):
    return "rgb(0, 128, 0)"  

wordcloud_pos = WordCloud(
    width=800,
    height=400,
    background_color="white",
    color_func=green_color_func
).generate_from_frequencies(dict(top_positivos.values))

wordcloud_neg = WordCloud(
    width=800, height=400, background_color="white", colormap="Reds"
).generate_from_frequencies(dict(top_negativos.values))

# Criar figura com dois subplots lado a lado
plt.figure(figsize=(18, 7))


# Word cloud positiva
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud – Positive Aspects", fontsize=16)

# Word cloud negativa
plt.subplot(1, 2, 2)
plt.imshow(wordcloud_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud – Negative Aspects", fontsize=16)

plt.tight_layout()
plt.savefig("graficos_llama/wordclouds_comparacao.png", dpi=300)

# =====================
# 5. Total comparison (All categories)
# =====================

apenas_positivos = df_final[(df_final["positivo"].str.lower() != "nenhum") & (df_final["negativo"].str.lower() == "nenhum")]
apenas_negativos = df_final[(df_final["positivo"].str.lower() == "nenhum") & (df_final["negativo"].str.lower() != "nenhum")]
ambos = df_final[(df_final["positivo"].str.lower() != "nenhum") & (df_final["negativo"].str.lower() != "nenhum")]

# Contagens adicionais
total_apenas_positivos = len(apenas_positivos)
total_apenas_negativos = len(apenas_negativos)
total_ambos = len(ambos)
total_nenhum = len(df_nenhum)
total_ofensivos = len(df_ofensivos)

# Dados consolidados
dados_gerais = pd.DataFrame({
    "category": [
        "Only Positive",
        "Only Negative",
        "Positive and Negative",
        "No Aspects",
        "Offensive Language"
    ],
    "count": [
        total_apenas_positivos,
        total_apenas_negativos,
        total_ambos,
        total_nenhum,
        total_ofensivos
    ]
})

# Paleta personalizada
cores = {
    "Only Positive": "green",
    "Only Negative": "red",
    "Positive and Negative": "goldenrod",
    "No Aspects": "gray",
    "Offensive Language": "purple"
}

# Gráfico
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=dados_gerais,
    x="category",
    y="count",
    hue="category",
    palette=cores,
    legend=False
)

# Adiciona o valor acima de cada barra
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f'{int(height)}',
        (p.get_x() + p.get_width() / 2, height),
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.title("Distribution of Comments by Sentiment and Category", fontsize=14)
plt.ylabel("Number of Comments")
plt.xlabel("")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("graficos_llama/total_distribution_detailed.png", dpi=300)
plt.show()
