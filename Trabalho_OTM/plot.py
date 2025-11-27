import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import os

# --- FUNÇÃO 1: PIZZA (MANTIDA IGUAL) ---
def plot_pizza_por_ativos(serie_pesos, risco, retorno, valor_investido, nome_arquivo, titulo_personalizado):
    # Filtra pesos irrelevantes
    pesos_relevantes = serie_pesos[serie_pesos > 0.0001]
    final_series = pesos_relevantes.sort_values(ascending=False)

    if final_series.empty:
        return

    labels = list(final_series.index)
    sizes = list(final_series.values)

    fig, ax = plt.subplots(figsize=(10, 8))
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
    plt.savefig(nome_arquivo, dpi=150)
    plt.close(fig)

# --- FUNÇÃO 2: BACKTEST (NOVA) ---
def plotar_comparativo_backtest(retornos_ativos, pesos_otimos, df_benchmarks, nome_arquivo):
    print(f"Gerando gráfico de Backtest: {nome_arquivo}")
    
    # 1. Calcular retorno diário da carteira (Soma Ponderada)
    retorno_diario_port = (retornos_ativos * pesos_otimos).sum(axis=1)
    
    # 2. Acumular (Base 100)
    carteira_acumulada = (1 + retorno_diario_port).cumprod()
    carteira_acumulada.name = "Minha Carteira"
    
    # 3. Juntar com Benchmarks
    df_final = pd.concat([carteira_acumulada, df_benchmarks], axis=1).dropna()
    
    # Normaliza tudo para base 100
    df_final = df_final / df_final.iloc[0] * 100
    
    # 4. Plotagem
    plt.figure(figsize=(10, 6))
    
    # Minha Carteira (Verde Forte)
    plt.plot(df_final.index, df_final['Minha Carteira'], 
             color='#006400', linewidth=2.5, label='Sua Carteira Otimizada')
    
    # Benchmarks
    if 'CDI' in df_final.columns:
        plt.plot(df_final.index, df_final['CDI'], 
                 color='black', linestyle='--', linewidth=1.5, label='CDI (Renda Fixa)')
    
    if 'Ibovespa' in df_final.columns:
        plt.plot(df_final.index, df_final['Ibovespa'], 
                 color='blue', alpha=0.5, linewidth=1, label='Ibovespa')
                 
    if 'S&P500 (BRL)' in df_final.columns:
        plt.plot(df_final.index, df_final['S&P500 (BRL)'], 
                 color='red', alpha=0.5, linewidth=1, label='S&P 500 (BRL)')
    
    plt.title("Comparativo Histórico (Simulação 5 Anos)", fontsize=14)
    plt.ylabel("Evolução de R$ 100", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150)
    plt.close()

# --- FUNÇÃO PRINCIPAL (AGORA GERA OS 4 GRÁFICOS) ---
def rodar_visualizacao_completa(inputs, res_ga, res_gurobi_warm, res_gurobi_cold, 
                                path_ga, path_gu_warm, path_gu_cold, path_backtest):
    
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
            
            plotar_comparativo_backtest(
                retornos_ativos=inputs['retornos_diarios_historicos'],
                pesos_otimos=pesos_array,
                df_benchmarks=inputs['df_benchmarks'],
                nome_arquivo=path_backtest
            )

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