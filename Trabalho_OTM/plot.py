import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import os

# Removida a variável global fixa para permitir nomes dinâmicos
def plot_pizza_por_ativos(serie_pesos, risco, retorno, valor_investido, nome_arquivo, titulo_personalizado):
    print(f"Gerando gráfico: {titulo_personalizado}")
    
    # Filtra pesos irrelevantes
    pesos_relevantes = serie_pesos[serie_pesos > 0.0001]
    final_series = pesos_relevantes.sort_values(ascending=False)

    if final_series.empty:
        print(f"Aviso: Carteira vazia para {titulo_personalizado}")
        return

    labels = list(final_series.index)
    sizes = list(final_series.values)

    fig, ax = plt.subplots(figsize=(10, 8)) # Tamanho levemente ajustado
    cmap = plt.get_cmap("nipy_spectral")
    colors = cmap(np.linspace(0.05, 0.95, len(labels)))

    def autopct_func(pct):
        return f"{pct:.1f}%" if pct >= 1.5 else ""

    wedges, texts, autotexts = ax.pie(
        sizes, 
        autopct=autopct_func,
        startangle=90, 
        colors=colors,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='w', linewidth=1)
    )

    plt.setp(autotexts, size=9, weight="bold", color="white", 
             path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
    
    ax.axis('equal')

    # Legenda Top 20
    num_legend = min(20, len(labels))
    plt.legend(
        wedges[:num_legend], labels[:num_legend],
        title=f"Top {num_legend} Ativos",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=9
    )
    
    plt.title(
        f"{titulo_personalizado}\n"
        f"Risco: {risco:.2%} | Retorno: {retorno:.2%}",
        fontsize=14, pad=20
    )
    
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150) # DPI ajustado para web
    print(f"Gráfico salvo em: '{nome_arquivo}'")
    plt.close(fig) # Importante para liberar memória

def rodar_visualizacao_dupla(inputs, res_ga, res_gurobi, caminho_img_ga, caminho_img_gurobi):
    """
    Gera dois gráficos comparativos.
    """
    valor_total = inputs['valor_total_investido']
    nomes_ativos = inputs['nomes_dos_ativos']

    # --- PLOT 1: ALGORITMO GENÉTICO ---
    if res_ga:
        # O GA já retorna um DataFrame pronto, pegamos a primeira linha
        row_ga = res_ga['dataframe_resultado'].iloc[0]
        cols_meta = ['Risco_Alvo', 'Risco_Encontrado_Anual', 'Retorno_Encontrado_Anual']
        pesos_ga = row_ga.drop(cols_meta, errors='ignore')
        
        plot_pizza_por_ativos(
            serie_pesos=pesos_ga,
            risco=row_ga['Risco_Encontrado_Anual'],
            retorno=row_ga['Retorno_Encontrado_Anual'],
            valor_investido=valor_total,
            nome_arquivo=caminho_img_ga,
            titulo_personalizado="Algoritmo Genético (Aproximação)"
        )

    # --- PLOT 2: GUROBI ---
    if res_gurobi:
        # O Gurobi retorna um dict com 'pesos' em numpy array. Precisamos converter para Series.
        pesos_gurobi_series = pd.Series(res_gurobi['pesos'], index=nomes_ativos)
        
        plot_pizza_por_ativos(
            serie_pesos=pesos_gurobi_series,
            risco=res_gurobi['risco'],
            retorno=res_gurobi['retorno'],
            valor_investido=valor_total,
            nome_arquivo=caminho_img_gurobi,
            titulo_personalizado="Solver Exato (Gurobi)"
        )