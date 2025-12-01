from flask import Flask, render_template, request, jsonify, url_for
import os
import time
import traceback
import numpy as np
import pandas as pd
import threading

# --- CONFIGURAÇÃO DE CAMINHOS ---
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

# ==============================================================================
# SISTEMA DE CACHE
# ==============================================================================
CACHE_DADOS = None
CACHE_LOCK = threading.Lock()
STATUS_CARREGAMENTO = "Aguardando..."

def tarefa_background_download():
    global CACHE_DADOS, STATUS_CARREGAMENTO
    with CACHE_LOCK:
        if CACHE_DADOS is not None:
            STATUS_CARREGAMENTO = "Dados Prontos"
            return
        print("--- [BACKGROUND] Iniciando pré-carregamento... ---")
        STATUS_CARREGAMENTO = "Baixando Ativos..."
        try:
            dados = preparar_dados.calcular_inputs_otimizacao(10000)
            if dados:
                CACHE_DADOS = dados
                STATUS_CARREGAMENTO = "Dados Prontos"
                print("--- [BACKGROUND] Dados carregados! ---")
            else:
                STATUS_CARREGAMENTO = "Erro no Download"
        except Exception as e:
            print(f"--- [BACKGROUND] Erro: {e}")
            STATUS_CARREGAMENTO = "Erro"

@app.route('/pre-carregar', methods=['GET'])
def trigger_pre_load():
    thread = threading.Thread(target=tarefa_background_download)
    thread.daemon = True
    thread.start()
    return jsonify({'status': 'iniciado'})

@app.route('/status-dados', methods=['GET'])
def check_status():
    return jsonify({'status': STATUS_CARREGAMENTO})

# ==============================================================================
# ROTAS PRINCIPAIS
# ==============================================================================

@app.route('/')
def index():
    mapa_setores = config.obter_mapa_setores_ativos()
    lista_setores = list(mapa_setores.keys())
    return render_template('index.html', setores=lista_setores)

# --- FUNÇÕES AUXILIARES ---

def safe_num(val):
    if val is None: return None
    try:
        if isinstance(val, (int, float, np.number)):
            if np.isnan(val) or np.isinf(val): return None
    except: pass
    return val

# --- FORMATAR DADOS (AGORA COM QUANTIDADE DE COTAS) ---
def formatar_dados_para_frontend(ticker_list, peso_array, valor_investido, precos_map, lotes_exatos=None):
    alocacao = []
    peso_array = np.nan_to_num(peso_array, nan=0.0)
    
    for i, ticker in enumerate(ticker_list):
        peso = float(peso_array[i])
        if peso > 1e-4:
            # Se vier do Gurobi, usa o lote exato. Se vier do GA, estima.
            if lotes_exatos is not None:
                qtd = int(lotes_exatos[i])
            else:
                preco = precos_map.get(ticker, 0.0)
                if preco > 0:
                    qtd = int((peso * valor_investido) / preco)
                else:
                    qtd = 0
            
            alocacao.append({
                'ativo': ticker,
                'qtd': qtd, # Campo novo
                'peso': round(peso * 100, 2),
                'valor': round(peso * valor_investido, 2)
            })
            
    soma_pesos = np.sum(peso_array)
    if soma_pesos < 0.999:
            sobra_peso = 1.0 - soma_pesos
            alocacao.append({
                'ativo': '🟢 CAIXA / NÃO INVESTIDO',
                'qtd': '-',
                'peso': round(sobra_peso * 100, 2),
                'valor': round(sobra_peso * valor_investido, 2)
            })
    return sorted(alocacao, key=lambda x: x['peso'], reverse=True)

# --- CÁLCULO DE SETORES (AGORA COM SOMA DE COTAS) ---
def calcular_alocacao_setorial(nomes_ativos, pesos_array, valor_investido, precos_map, lotes_exatos=None):
    mapa_setores = config.obter_mapa_setores_ativos()
    
    ativo_para_setor = {}
    for setor, lista_ativos in mapa_setores.items():
        for ativo in lista_ativos:
            ativo_para_setor[ativo] = setor
            
    totais_setor = {} # {Setor: Peso}
    totais_qtd_setor = {} # {Setor: Qtd}
    
    pesos_array = np.nan_to_num(pesos_array, nan=0.0)
    
    for i, ativo in enumerate(nomes_ativos):
        peso = float(pesos_array[i])
        if peso > 1e-4:
            nome_setor = ativo_para_setor.get(ativo, "Outros")
            
            if nome_setor not in totais_setor:
                totais_setor[nome_setor] = 0.0
                totais_qtd_setor[nome_setor] = 0
                
            totais_setor[nome_setor] += peso
            
            # Cálculo da Qtd
            if lotes_exatos is not None:
                qtd = int(lotes_exatos[i])
            else:
                preco = precos_map.get(ativo, 0.0)
                qtd = int((peso * valor_investido) / preco) if preco > 0 else 0
                
            totais_qtd_setor[nome_setor] += qtd
            
    soma_total = np.sum(pesos_array)
    if soma_total < 0.999:
        sobra = 1.0 - soma_total
        totais_setor['🟢 CAIXA / NÃO INVESTIDO'] = sobra
        totais_qtd_setor['🟢 CAIXA / NÃO INVESTIDO'] = 0
        
    resultado = []
    for setor, peso_total in totais_setor.items():
        resultado.append({
            'setor': setor,
            'qtd': totais_qtd_setor.get(setor, 0), # Campo novo
            'peso': round(peso_total * 100, 2),
            'valor': round(peso_total * valor_investido, 2)
        })
        
    return sorted(resultado, key=lambda x: x['peso'], reverse=True)

def contar_ativos_setores(pesos_array, alocacao_setorial_lista):
    qtd_ativos = np.sum(np.nan_to_num(pesos_array) > 1e-4)
    qtd_setores = len([s for s in alocacao_setorial_lista if "CAIXA" not in s['setor']])
    return int(qtd_ativos), int(qtd_setores)

@app.route('/otimizar', methods=['POST'])
def processar_otimizacao():
    global CACHE_DADOS
    try:
        dados = request.json
        valor_investir = float(dados.get('valor') or 0)
        
        lambda_risco = float(dados.get('lambda') or 50.0)
        risco_teto = float(dados.get('risco') or 15) / 100.0
        teto_ativo_input = float(dados.get('teto_ativo') or 30.0) / 100.0
        teto_setor_input = float(dados.get('teto_setor') or 100.0) / 100.0
        setores_proibidos = dados.get('proibidos', [])
        
        max_ativos_global = int(dados.get('max_ativos') or 15)
        max_ativos_por_setor = int(dados.get('max_ativos_setor') or 4)
        
        print(f"\n--- [POST /otimizar] Iniciando... ---")
        
        inputs = None
        with CACHE_LOCK:
            if CACHE_DADOS is not None:
                inputs = CACHE_DADOS.copy()
                inputs['valor_total_investido'] = valor_investir
        
        if inputs is None:
            inputs = preparar_dados.calcular_inputs_otimizacao(valor_investir)
            with CACHE_LOCK: CACHE_DADOS = inputs
        
        if inputs is None:
            return jsonify({'sucesso': False, 'erro': 'Falha ao baixar dados.'}), 500
        
        nomes_ativos = inputs['nomes_dos_ativos']
        
        # Pega os preços para calcular as quantidades
        precos_map = inputs.get('ultimos_precos', pd.Series()).to_dict()

        # 1. GA
        print(">> Rodando GA...")
        start_ga = time.time()
        res_ga = otimizar.rodar_otimização(inputs, risco_teto, lambda_risco, setores_proibidos, 
                                           teto_maximo_ativo=teto_ativo_input, 
                                           teto_maximo_setor=teto_setor_input)
        tempo_ga = time.time() - start_ga
        
        if res_ga is None: return jsonify({'sucesso': False, 'erro': 'GA não convergiu.'}), 400

        # 2. Gurobi Warm
        print(">> Rodando Gurobi (Warm)...")
        start_gu_warm = time.time()
        res_gurobi_warm = otm_gurobi_setores.resolver_com_gurobi_setores(
            inputs, lambda_risco, risco_teto, 
            warm_start_pesos=res_ga['pesos_finais'], setores_proibidos=setores_proibidos, 
            teto_maximo_ativo=teto_ativo_input, teto_maximo_setor=teto_setor_input,
            max_ativos_carteira=max_ativos_global,
            max_ativos_setor=max_ativos_por_setor
        )
        tempo_gu_warm = time.time() - start_gu_warm

        # 3. Gurobi Cold
        print(">> Rodando Gurobi (Cold)...")
        start_gu_cold = time.time()
        res_gurobi_cold = otm_gurobi_setores.resolver_com_gurobi_setores(
            inputs, lambda_risco, risco_teto, 
            warm_start_pesos=None, setores_proibidos=setores_proibidos, 
            teto_maximo_ativo=teto_ativo_input, teto_maximo_setor=teto_setor_input,
            max_ativos_carteira=max_ativos_global,
            max_ativos_setor=max_ativos_por_setor
        )
        tempo_gu_cold = time.time() - start_gu_cold

        # 4. Gráficos
        timestamp = int(time.time())
        nome_ga, nome_gu_warm, nome_gu_cold = 'grafico_ga.png', 'grafico_gurobi_warm.png', 'grafico_gurobi_cold.png'
        
        plot.rodar_visualizacao_completa(
            inputs, res_ga, res_gurobi_warm, res_gurobi_cold, 
            os.path.join(STATIC_DIR, nome_ga), 
            os.path.join(STATIC_DIR, nome_gu_warm), 
            os.path.join(STATIC_DIR, nome_gu_cold)
        )

        # 5. Dados Interativos
        retornos_hist = inputs['retornos_diarios_historicos']
        df_bench = inputs['df_benchmarks']
        
        def clean_list(lst): return [safe_num(x) for x in lst]
        
        try: bench_cdi = clean_list((df_bench['CDI'] * 100).fillna(100).tolist())
        except: bench_cdi = []
        try: bench_ibov = clean_list((df_bench['Ibovespa'] * 100).fillna(100).tolist())
        except: bench_ibov = []
        try: bench_sp500 = clean_list((df_bench['S&P500 (BRL)'] * 100).fillna(100).tolist())
        except: bench_sp500 = []

        def preparar_backtest_carteira(pesos, nomes):
            dict_pesos = dict(zip(nomes, pesos))
            pesos_ordenados = [dict_pesos.get(col, 0.0) for col in retornos_hist.columns]
            datas_cart, valores_cart = preparar_dados.simular_evolucao_diaria(
                retornos_hist, pesos_ordenados, valor_inicial=100
            )
            return datas_cart, clean_list(valores_cart)

        # --- A. DADOS GA ---
        row_ga = res_ga['dataframe_resultado'].iloc[0]
        pesos_ga = row_ga.drop(['Risco_Alvo', 'Risco_Encontrado_Anual', 'Retorno_Encontrado_Anual'], errors='ignore')
        pesos_ga_final = pesos_ga.reindex(nomes_ativos).fillna(0.0).values
        datas_ga, valores_ga = preparar_backtest_carteira(pesos_ga_final, nomes_ativos)
        
        # GA: Passamos None nos lotes_exatos, então ele calcula baseado no peso
        aloc_setor_ga = calcular_alocacao_setorial(nomes_ativos, pesos_ga_final, valor_investir, precos_map)
        n_ativos_ga, n_setores_ga = contar_ativos_setores(pesos_ga_final, aloc_setor_ga)

        data_ga = {
            'metricas': {
                'valor_investido': safe_num(valor_investir),
                'retorno_aa': safe_num(res_ga['retorno_final'] * 100),
                'risco_aa': safe_num(res_ga['risco_final'] * 100),
                'score': safe_num(res_ga['funcao_objetivo']),
                'tempo': safe_num(tempo_ga),
                'pvp': safe_num(res_ga['metricas'].get('pvp_final')),
                'cvar': safe_num(res_ga['metricas'].get('cvar_final', 0) * 100),
                'qtd_ativos': n_ativos_ga,
                'qtd_setores': n_setores_ga
            },
            'alocacao': formatar_dados_para_frontend(nomes_ativos, pesos_ga_final, valor_investir, precos_map),
            'alocacao_setorial': aloc_setor_ga,
            'grafico_url': url_for('static', filename=nome_ga) + f'?t={timestamp}',
            'backtest': {'datas': datas_ga, 'carteira': valores_ga, 'cdi': bench_cdi, 'ibov': bench_ibov, 'sp500': bench_sp500}
        }

        # --- B. DADOS GUROBI WARM ---
        data_gu_warm = None
        if res_gurobi_warm:
            datas_gu, valores_gu = preparar_backtest_carteira(res_gurobi_warm['pesos'], nomes_ativos)
            
            # Gurobi: Passamos res_gurobi_warm['lotes'] para ter a qtd exata
            lotes_warm = res_gurobi_warm.get('lotes') 
            
            aloc_setor_warm = calcular_alocacao_setorial(nomes_ativos, res_gurobi_warm['pesos'], valor_investir, precos_map, lotes_warm)
            n_ativos_warm, n_setores_warm = contar_ativos_setores(res_gurobi_warm['pesos'], aloc_setor_warm)

            data_gu_warm = {
                'metricas': {
                    'valor_investido': safe_num(valor_investir),
                    'retorno_aa': safe_num(res_gurobi_warm['retorno'] * 100),
                    'risco_aa': safe_num(res_gurobi_warm['risco'] * 100),
                    'score': safe_num(res_gurobi_warm['obj']),
                    'tempo': safe_num(tempo_gu_warm),
                    'pvp': safe_num(res_gurobi_warm.get('pvp_final')),
                    'cvar': safe_num(res_gurobi_warm.get('cvar_final', 0) * 100),
                    'qtd_ativos': n_ativos_warm,
                    'qtd_setores': n_setores_warm
                },
                'alocacao': formatar_dados_para_frontend(nomes_ativos, res_gurobi_warm['pesos'], valor_investir, precos_map, lotes_warm),
                'alocacao_setorial': aloc_setor_warm,
                'grafico_url': url_for('static', filename=nome_gu_warm) + f'?t={timestamp}',
                'backtest': {'datas': datas_gu, 'carteira': valores_gu, 'cdi': bench_cdi, 'ibov': bench_ibov, 'sp500': bench_sp500}
            }

        # --- C. DADOS GUROBI COLD ---
        data_gu_cold = None
        if res_gurobi_cold:
            datas_cold, valores_cold = preparar_backtest_carteira(res_gurobi_cold['pesos'], nomes_ativos)
            
            lotes_cold = res_gurobi_cold.get('lotes')
            
            aloc_setor_cold = calcular_alocacao_setorial(nomes_ativos, res_gurobi_cold['pesos'], valor_investir, precos_map, lotes_cold)
            n_ativos_cold, n_setores_cold = contar_ativos_setores(res_gurobi_cold['pesos'], aloc_setor_cold)

            data_gu_cold = {
                'metricas': {
                    'valor_investido': safe_num(valor_investir),
                    'retorno_aa': safe_num(res_gurobi_cold['retorno'] * 100),
                    'risco_aa': safe_num(res_gurobi_cold['risco'] * 100),
                    'score': safe_num(res_gurobi_cold['obj']),
                    'tempo': safe_num(tempo_gu_cold),
                    'pvp': safe_num(res_gurobi_cold.get('pvp_final')),
                    'cvar': safe_num(res_gurobi_cold.get('cvar_final', 0) * 100),
                    'qtd_ativos': n_ativos_cold,
                    'qtd_setores': n_setores_cold
                },
                'alocacao': formatar_dados_para_frontend(nomes_ativos, res_gurobi_cold['pesos'], valor_investir, precos_map, lotes_cold),
                'alocacao_setorial': aloc_setor_cold,
                'grafico_url': url_for('static', filename=nome_gu_cold) + f'?t={timestamp}',
                'backtest': {'datas': datas_cold, 'carteira': valores_cold, 'cdi': bench_cdi, 'ibov': bench_ibov, 'sp500': bench_sp500}
            }

        return jsonify({'sucesso': True, 'ga': data_ga, 'gurobi_warm': data_gu_warm, 'gurobi_cold': data_gu_cold})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'sucesso': False, 'erro': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)