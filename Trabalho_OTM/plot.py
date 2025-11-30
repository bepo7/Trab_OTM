import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import os

# --- FUNÇÃO 1: PIZZA (MANTIDA IGUAL) ---
def plot_pizza_por_ativos(serie_pesos, risco, retorno, valor_investido, nome_arquivo, titulo_personalizado):
    # 1. Verifica sobra de caixa
    soma_pesos = serie_pesos.sum()
    if soma_pesos < 0.999:
        sobra = 1.0 - soma_pesos
        serie_pesos["Caixa / Não Investido"] = sobra

    # Filtra pesos irrelevantes
    pesos_relevantes = serie_pesos[serie_pesos > 0.0001]
    final_series = pesos_relevantes.sort_values(ascending=False)

    if final_series.empty:
        return

    labels = list(final_series.index)
    sizes = list(final_series.values)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Cores: Gera mapa espectral para ativos, e Cinza para Caixa
    cmap = plt.get_cmap("nipy_spectral")
    colors = []
    num_ativos = len(labels)
    
    # Se tiver caixa, ele entra na conta das cores, mas forçamos cinza
    idx_cor = 0
    total_ativos_reais = len([l for l in labels if l != "Caixa / Não Investido"])
    
    for label in labels:
        if label == "Caixa / Não Investido":
            colors.append("#D3D3D3") # Cinza claro
        else:
            # Distribui as cores espectrais apenas entre os ativos reais
            ratio = idx_cor / max(1, total_ativos_reais - 1)
            colors.append(cmap(0.05 + 0.9 * ratio))
            idx_cor += 1

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
        f"Volatilidade: {risco:.2%} | Retorno: {retorno:.2%}",
        fontsize=14, pad=20
    )
    
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150)
    plt.close(fig)

# --- FUNÇÃO PRINCIPAL (AGORA GERA OS 4 GRÁFICOS) ---
def rodar_visualizacao_completa(inputs, res_ga, res_gurobi_warm, res_gurobi_cold, 
                                path_ga, path_gu_warm, path_gu_cold):
    
    valor_total = inputs['valor_total_investido']
    nomes_ativos = inputs['nomes_dos_ativos']

    # 1. PIZZA GA
    if res_ga:
        row_ga = res_ga['dataframe_resultado'].iloc[0]
        # Dropa colunas que não são pesos
        cols_meta = ['Risco_Alvo', 'Risco_Encontrado_Anual', 'Retorno_Encontrado_Anual']
        pesos_ga = row_ga.drop(cols_meta, errors='ignore')
        
        plot_pizza_por_ativos(
            serie_pesos=pesos_ga,
            risco=res_ga['risco_final'],
            retorno=res_ga['retorno_final'],
            valor_investido=valor_total,
            nome_arquivo=path_ga,
            titulo_personalizado="Algoritmo Genético"
        )
        
        # 1.1 GERAR BACKTEST (Usando pesos do GA como referência principal)
        # Se quiser usar os pesos do Gurobi, basta trocar aqui
        if 'retornos_diarios_historicos' in inputs and 'df_benchmarks' in inputs:
            # Transforma pesos_ga (Series) em array numpy para cálculo
            pesos_array = pesos_ga.reindex(nomes_ativos).fillna(0).values
    

    # 2. PIZZA GUROBI WARM
    if res_gurobi_warm:
        pesos_warm = pd.Series(res_gurobi_warm['pesos'], index=nomes_ativos)
        plot_pizza_por_ativos(
            serie_pesos=pesos_warm,
            risco=res_gurobi_warm['risco'],
            retorno=res_gurobi_warm['retorno'],
            valor_investido=valor_total,
            nome_arquivo=path_gu_warm,
            titulo_personalizado="Gurobi (Warm Start)"
        )

    # 3. PIZZA GUROBI COLD
    if res_gurobi_cold:
        pesos_cold = pd.Series(res_gurobi_cold['pesos'], index=nomes_ativos)
        plot_pizza_por_ativos(
            serie_pesos=pesos_cold,
            risco=res_gurobi_cold['risco'],
            retorno=res_gurobi_cold['retorno'],
            valor_investido=valor_total,
            nome_arquivo=path_gu_cold,
            titulo_personalizado="Gurobi (Cold Start)"
        )