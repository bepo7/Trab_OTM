from flask import Flask, render_template, request, jsonify, url_for
import os
import time
import traceback
import numpy as np
import pandas as pd

# --- CONFIGURAÇÃO DE CAMINHOS ABSOLUTOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'plots')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'site')

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATE_DIR)

import config
import preparar_dados
import otimizar
import otm_gurobi_setores
import plot

@app.route('/')
def index():
    mapa_setores = config.obter_mapa_setores_ativos()
    lista_setores = list(mapa_setores.keys())
    return render_template('index.html', setores=lista_setores)

def formatar_dados_para_frontend(ticker_list, peso_array, valor_investido):
    alocacao = []
    for i, ticker in enumerate(ticker_list):
        peso = peso_array[i]
        if peso > 1e-4: 
            alocacao.append({
                'ativo': ticker,
                'peso': round(peso * 100, 2),
                'valor': round(peso * valor_investido, 2)
            })
    return sorted(alocacao, key=lambda x: x['peso'], reverse=True)

@app.route('/otimizar', methods=['POST'])
def processar_otimizacao():
    try:
        dados = request.json
        valor_investir = float(dados.get('valor') or 0)
        risco_teto = float(dados.get('risco') or 15) / 100.0
        lambda_risco = float(dados.get('lambda') or 50.0) 
        setores_proibidos = dados.get('proibidos', [])

        print(f"\n--- [POST /otimizar] Iniciando: Risco={risco_teto:.1%}, Lambda={lambda_risco} ---")
        
        # 1. Preparar Dados
        inputs = preparar_dados.calcular_inputs_otimizacao(valor_investir)
        if inputs is None:
            return jsonify({'sucesso': False, 'erro': 'Falha ao baixar dados.'}), 500
        
        nomes_ativos = inputs['nomes_dos_ativos']

        # ---------------------------------------------------------
        # 2. Execução: ALGORITMO GENÉTICO (Base)
        # ---------------------------------------------------------
        print(">> Rodando GA...")
        start_ga = time.time()
        res_ga = otimizar.rodar_otimização(inputs, risco_teto, lambda_risco, setores_proibidos)
        tempo_ga = time.time() - start_ga
        
        if res_ga is None:
            return jsonify({'sucesso': False, 'erro': 'GA não convergiu.'}), 400

        # ---------------------------------------------------------
        # 3a. Execução: GUROBI COM WARM START (Ajuda do GA)
        # ---------------------------------------------------------
        print(">> Rodando Gurobi (Com GA)...")
        pesos_iniciais_ga = res_ga['pesos_finais']
        start_gu_warm = time.time()
        res_gurobi_warm = otm_gurobi_setores.resolver_com_gurobi_setores(
            inputs, lambda_risco, risco_teto, 
            warm_start_pesos=pesos_iniciais_ga, 
            setores_proibidos=setores_proibidos
        )
        tempo_gu_warm = time.time() - start_gu_warm

        # ---------------------------------------------------------
        # 3b. Execução: GUROBI COLD START (Sem GA)
        # ---------------------------------------------------------
        print(">> Rodando Gurobi (Puro/Cold Start)...")
        start_gu_cold = time.time()
        res_gurobi_cold = otm_gurobi_setores.resolver_com_gurobi_setores(
            inputs, lambda_risco, risco_teto, 
            warm_start_pesos=None,  
            setores_proibidos=setores_proibidos
        )
        tempo_gu_cold = time.time() - start_gu_cold

        # 4. Geração de Imagens
        timestamp = int(time.time())
        nome_ga = 'grafico_ga.png'
        nome_gu_warm = 'grafico_gurobi_warm.png'
        nome_gu_cold = 'grafico_gurobi_cold.png'
        nome_backtest = 'grafico_backtest.png'
        
        plot.rodar_visualizacao_completa(
            inputs, res_ga, res_gurobi_warm, res_gurobi_cold, 
            os.path.join(STATIC_DIR, nome_ga), 
            os.path.join(STATIC_DIR, nome_gu_warm), 
            os.path.join(STATIC_DIR, nome_gu_cold), 
            os.path.join(STATIC_DIR, nome_backtest)
        )

       # ---------------------------------------------------------
        # 5. PREPARAR DADOS PARA O GRÁFICO INTERATIVO (Chart.js)
        # ---------------------------------------------------------
        retornos_hist = inputs['retornos_diarios_historicos']
        df_bench = inputs['df_benchmarks']
        
        # Preparar Benchmarks (CDI, IBOV, S&P500) - Normaliza para Base 100
        # O try/except é para evitar crash se uma coluna falhar
        try: bench_cdi = (df_bench['CDI'] * 100).fillna(100).tolist()
        except: bench_cdi = []
        
        try: bench_ibov = (df_bench['Ibovespa'] * 100).fillna(100).tolist()
        except: bench_ibov = []
            
        try: bench_sp500 = (df_bench['S&P500 (BRL)'] * 100).fillna(100).tolist()
        except: bench_sp500 = []

        def preparar_backtest_carteira(pesos, nomes):
            dict_pesos = dict(zip(nomes, pesos))
            pesos_ordenados = [dict_pesos.get(col, 0.0) for col in retornos_hist.columns]
            
            # Base 100 para ficar na mesma escala dos benchmarks
            datas_cart, valores_cart = preparar_dados.simular_evolucao_diaria(
                retornos_hist, pesos_ordenados, valor_inicial=100
            )
            
            # Limpeza de NaN para JSON
            valores_limpos = [None if pd.isna(x) else x for x in valores_cart]
            return datas_cart, valores_limpos

        # --- A. DADOS GA ---
        row_ga = res_ga['dataframe_resultado'].iloc[0]
        cols_meta = ['Risco_Alvo', 'Risco_Encontrado_Anual', 'Retorno_Encontrado_Anual']
        pesos_ga_series = row_ga.drop(cols_meta, errors='ignore')
        pesos_ga_final = pesos_ga_series.reindex(nomes_ativos).fillna(0.0).values
        
        datas_ga, valores_ga = preparar_backtest_carteira(pesos_ga_final, nomes_ativos)

        data_ga = {
            'metricas': {
                'valor_investido': valor_investir,
                'retorno_aa': res_ga['retorno_final'] * 100,
                'risco_aa': res_ga['risco_final'] * 100,
                'score': res_ga['funcao_objetivo'],
                'tempo': tempo_ga
            },
            'alocacao': formatar_dados_para_frontend(nomes_ativos, pesos_ga_final, valor_investir),
            'grafico_url': url_for('static', filename=nome_ga) + f'?t={timestamp}',
            'backtest': {
                'datas': datas_ga,
                'carteira': valores_ga,
                'cdi': bench_cdi,
                'ibov': bench_ibov,
                'sp500': bench_sp500 # <--- CAMPO NOVO
            }
        }

        # --- B. DADOS GUROBI WARM ---
        data_gu_warm = None
        if res_gurobi_warm:
            datas_gu, valores_gu = preparar_backtest_carteira(res_gurobi_warm['pesos'], nomes_ativos)
            data_gu_warm = {
                'metricas': {
                    'valor_investido': valor_investir,
                    'retorno_aa': res_gurobi_warm['retorno'] * 100,
                    'risco_aa': res_gurobi_warm['risco'] * 100,
                    'score': res_gurobi_warm['obj'],
                    'tempo': tempo_gu_warm
                },
                'alocacao': formatar_dados_para_frontend(nomes_ativos, res_gurobi_warm['pesos'], valor_investir),
                'grafico_url': url_for('static', filename=nome_gu_warm) + f'?t={timestamp}',
                'backtest': {
                    'datas': datas_gu,
                    'carteira': valores_gu,
                    'cdi': bench_cdi,
                    'ibov': bench_ibov,
                    'sp500': bench_sp500 # <--- CAMPO NOVO
                }
            }

        # --- C. DADOS GUROBI COLD ---
        data_gu_cold = None
        if res_gurobi_cold:
            datas_cold, valores_cold = preparar_backtest_carteira(res_gurobi_cold['pesos'], nomes_ativos)
            data_gu_cold = {
                'metricas': {
                    'valor_investido': valor_investir,
                    'retorno_aa': res_gurobi_cold['retorno'] * 100,
                    'risco_aa': res_gurobi_cold['risco'] * 100,
                    'score': res_gurobi_cold['obj'],
                    'tempo': tempo_gu_cold
                },
                'alocacao': formatar_dados_para_frontend(nomes_ativos, res_gurobi_cold['pesos'], valor_investir),
                'grafico_url': url_for('static', filename=nome_gu_cold) + f'?t={timestamp}',
                'backtest': {
                    'datas': datas_cold,
                    'carteira': valores_cold,
                    'cdi': bench_cdi,
                    'ibov': bench_ibov,
                    'sp500': bench_sp500 # <--- CAMPO NOVO
                }
            }

        return jsonify({
            'sucesso': True,
            'ga': data_ga,
            'gurobi_warm': data_gu_warm,
            'gurobi_cold': data_gu_cold
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'sucesso': False, 'erro': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)