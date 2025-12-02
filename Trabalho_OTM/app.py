from flask import Flask, render_template, request, jsonify, url_for
import os
import time
import traceback
import numpy as np
import pandas as pd
import threading
import concurrent.futures
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Configuração dos diretórios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'plots')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'site')

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATE_DIR)


import config
import preparar_dados
import modelo_AG
import modelo_GUROBI
import plot


CACHE_DADOS = None
CACHE_LOCK = threading.Lock()
STATUS_CARREGAMENTO = "Aguardando..."

# Função de pré-carregamento em background
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


@app.route('/')
def index():
    mapa_setores = config.obter_mapa_setores_ativos()
    lista_setores = list(mapa_setores.keys())
    return render_template('index.html', setores=lista_setores)

@app.route('/grafico_temporal_<tipo>.png')
def serve_temporal_chart(tipo):
    """Serve temporal analysis chart images"""
    from flask import send_from_directory
    return send_from_directory(STATIC_DIR, f'grafico_temporal_{tipo}.png')

# Função segura para converter valores para float
def safe_num(val):
    if val is None: return None
    try:
        if isinstance(val, (int, float, np.number)):
            if np.isnan(val) or np.isinf(val): return None
    except: pass
    return val

# Função para formatar os dados de alocação para o front-end
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
                'qtd': qtd,
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

# Função para calcular alocação setorial
def calcular_alocacao_setorial(nomes_ativos, pesos_array, valor_investido, precos_map, lotes_exatos=None):
    mapa_setores = config.obter_mapa_setores_ativos()
    
    ativo_para_setor = {}
    for setor, lista_ativos in mapa_setores.items():
        for ativo in lista_ativos:
            ativo_para_setor[ativo] = setor
            
    totais_setor = {}
    totais_qtd_setor = {}
    
    pesos_array = np.nan_to_num(pesos_array, nan=0.0)
    
    # Cálculo dos totais por setor
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

# Função para contar ativos e setores
def contar_ativos_setores(pesos_array, alocacao_setorial_lista):
    qtd_ativos = np.sum(np.nan_to_num(pesos_array) > 1e-4)
    qtd_setores = len([s for s in alocacao_setorial_lista if "CAIXA" not in s['setor']])
    return int(qtd_ativos), int(qtd_setores)

@app.route('/otimizar', methods=['POST'])

# Função principal para processar a otimização
def processar_otimizacao():
    global CACHE_DADOS
    try:
        dados = request.json
        valor_investir = float(dados.get('valor') or 0)
        
        # Variáveis de controle
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

        # 1 - Algoritmo Genético
        print(">> Rodando GA...")
        start_ga = time.time()
        res_ga = modelo_AG.rodar_otimização(inputs, risco_teto, lambda_risco, setores_proibidos, 
                                           teto_maximo_ativo=teto_ativo_input, 
                                           teto_maximo_setor=teto_setor_input)
        tempo_ga = time.time() - start_ga
        
        if res_ga is None: return jsonify({'sucesso': False, 'erro': 'GA não convergiu.'}), 400

        # 2 - Gurobi Warm
        print(">> Rodando Gurobi (Warm)...")
        start_gu_warm = time.time()
        res_gurobi_warm = modelo_GUROBI.resolver_com_gurobi_setores(
            inputs, lambda_risco, risco_teto, 
            warm_start_pesos=res_ga['pesos_finais'], setores_proibidos=setores_proibidos, 
            teto_maximo_ativo=teto_ativo_input, teto_maximo_setor=teto_setor_input,
            max_ativos_carteira=max_ativos_global,
            max_ativos_setor=max_ativos_por_setor
        )
        tempo_gu_warm = time.time() - start_gu_warm

        # 3 - Gurobi Cold
        print(">> Rodando Gurobi (Cold)...")
        start_gu_cold = time.time()
        res_gurobi_cold = modelo_GUROBI.resolver_com_gurobi_setores(
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

        # Função para preparar backtest
        def preparar_backtest_carteira(pesos, nomes):
            dict_pesos = dict(zip(nomes, pesos))
            pesos_ordenados = [dict_pesos.get(col, 0.0) for col in retornos_hist.columns]
            datas_cart, valores_cart = preparar_dados.simular_evolucao_diaria(
                retornos_hist, pesos_ordenados, valor_inicial=100
            )
            return datas_cart, clean_list(valores_cart)

        # Dados do GA
        row_ga = res_ga['dataframe_resultado'].iloc[0]
        pesos_ga = row_ga.drop(['Risco_Alvo', 'Risco_Encontrado_Anual', 'Retorno_Encontrado_Anual'], errors='ignore')
        pesos_ga_final = pesos_ga.reindex(nomes_ativos).fillna(0.0).values
        datas_ga, valores_ga = preparar_backtest_carteira(pesos_ga_final, nomes_ativos)
        
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

        # Dados do Gurobi Warm
        data_gu_warm = None
        if res_gurobi_warm:
            datas_gu, valores_gu = preparar_backtest_carteira(res_gurobi_warm['pesos'], nomes_ativos)
            
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

        # Dados do Gurobi Cold
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

@app.route('/otimizar-temporal', methods=['POST'])

# Função principal para processar a otimização temporal
def processar_otimizacao_temporal():
  
    try:
        dados = request.json
        valor_investir = float(dados.get('valor') or 100000)
        
        # Variáveis de controle
        lambda_risco = float(dados.get('lambda') or 50.0)
        risco_teto = float(dados.get('risco') or 15) / 100.0
        teto_ativo_input = float(dados.get('teto_ativo') or 30.0) / 100.0
        teto_setor_input = float(dados.get('teto_setor') or 100.0) / 100.0
        setores_proibidos = dados.get('proibidos', [])
        
        max_ativos_global = int(dados.get('max_ativos') or 15)
        max_ativos_por_setor = int(dados.get('max_ativos_setor') or 4)
        
        print(f"\n{'='*80}")
        print(f"ANÁLISE TEMPORAL DE CARTEIRA")
        print(f"{'='*80}")
        
        print(f"\n[DOWNLOAD ÚNICO] Baixando dados de {config.DATA_INICIO_COMPLETO} a {config.DATA_FIM_COMPLETO}")
        
        inputs_completo = preparar_dados.calcular_inputs_otimizacao_periodo(
            valor_investir,
            config.DATA_INICIO_COMPLETO,
            config.DATA_FIM_COMPLETO
        )
        
        if inputs_completo is None:
            return jsonify({'sucesso': False, 'erro': 'Falha ao baixar dados (2021-2024).'}), 500
        
        # Fase 1: Otimização com dados de treino
        print(f"\n[FASE 1] Otimizando carteira com dados de {config.DATA_INICIO_TREINO} a {config.DATA_FIM_TREINO}")
        
        # Reutiliza os dados completos, filtrando para o período de treino
        inputs_treino = preparar_dados.calcular_inputs_otimizacao_periodo(
            valor_investir, 
            config.DATA_INICIO_TREINO, 
            config.DATA_FIM_TREINO
        )
        
        if inputs_treino is None:
            return jsonify({'sucesso': False, 'erro': 'Falha ao processar dados de treino (2021-2023).'}), 500
        
        nomes_ativos_treino = inputs_treino['nomes_dos_ativos']
        precos_map_treino = inputs_treino.get('ultimos_precos', pd.Series()).to_dict()
        
        # Otimiza com Gurobi (usando dados de treino)
        print("\n>> Otimizando com Gurobi (dados 2021-2022)...")
        print(f"   Ativos: {len(nomes_ativos_treino)}, Setores proibidos: {setores_proibidos}")
        print(f"   Parâmetros: max_ativos={max_ativos_global}, max_ativos_setor={max_ativos_por_setor}")
        start_treino = time.time()
        res_gurobi_treino = modelo_GUROBI.resolver_com_gurobi_setores(
            inputs_treino, lambda_risco, risco_teto,
            warm_start_pesos=None, setores_proibidos=setores_proibidos,
            teto_maximo_ativo=teto_ativo_input, teto_maximo_setor=teto_setor_input,
            max_ativos_carteira=max_ativos_global,
            max_ativos_setor=max_ativos_por_setor
        )
        tempo_treino = time.time() - start_treino
        
        if res_gurobi_treino is None:
            return jsonify({'sucesso': False, 'erro': 'Otimização de treino falhou.'}), 400
        
        pesos_treino = res_gurobi_treino['pesos']
        lotes_treino = res_gurobi_treino.get('lotes')
        
        # Métricas de treino
        aloc_setor_treino = calcular_alocacao_setorial(nomes_ativos_treino, pesos_treino, valor_investir, precos_map_treino, lotes_treino)
        n_ativos_treino, n_setores_treino = contar_ativos_setores(pesos_treino, aloc_setor_treino)
        
        metricas_treino = {
            'periodo': f"{config.DATA_INICIO_TREINO} a {config.DATA_FIM_TREINO}",
            'valor_investido': safe_num(valor_investir),
            'retorno_aa': safe_num(res_gurobi_treino['retorno'] * 100),
            'risco_aa': safe_num(res_gurobi_treino['risco'] * 100),
            'score': safe_num(res_gurobi_treino['obj']),
            'tempo': safe_num(tempo_treino),
            'pvp': safe_num(res_gurobi_treino.get('pvp_final')),
            'cvar': safe_num(res_gurobi_treino.get('cvar_final', 0) * 100),
            'qtd_ativos': n_ativos_treino,
            'qtd_setores': n_setores_treino
        }
        
        alocacao_treino = formatar_dados_para_frontend(nomes_ativos_treino, pesos_treino, valor_investir, precos_map_treino, lotes_treino)
        
        if n_ativos_treino == 0:
            return jsonify({'sucesso': False, 'erro': 'A otimização de treino resultou em uma carteira vazia (100% caixa). Tente reduzir a aversão ao risco ou aumentar a penalidade de caixa.'}), 400

        # Fase 2: Simulação da performance no período de teste
        print(f"\n[FASE 2] Simulando performance da carteira 2021-2022 no período {config.DATA_INICIO_TESTE} a {config.DATA_FIM_TESTE}")
        
        performance_teste = preparar_dados.simular_performance_periodo(
            pesos_treino,
            nomes_ativos_treino,
            config.DATA_INICIO_TESTE,
            config.DATA_FIM_TESTE,
            valor_inicial=valor_investir
        )
        
        if performance_teste is None:
            return jsonify({'sucesso': False, 'erro': 'Falha na simulação de performance 2023-2024 (nenhum ativo disponível).'}), 400
        
        metricas_teste = {
            'periodo': f"{config.DATA_INICIO_TESTE} a {config.DATA_FIM_TESTE}",
            'retorno_realizado_aa': safe_num(performance_teste['retorno_aa'] * 100),
            'risco_realizado_aa': safe_num(performance_teste['risco_aa'] * 100),
            'valor_final': safe_num(performance_teste['valor_final']),
            'retorno_total': safe_num(performance_teste['retorno_total'] * 100)
        }
        
        # Fase 3: Otimização com dados completos
        print(f"\n[FASE 3] Otimizando carteira com dados completos de {config.DATA_INICIO_COMPLETO} a {config.DATA_FIM_COMPLETO}")
        
        # Dados já foram baixados no início, apenas reutiliza
        nomes_ativos_completo = inputs_completo['nomes_dos_ativos']
        precos_map_completo = inputs_completo.get('ultimos_precos', pd.Series()).to_dict()
        
        # Otimiza com Gurobi (usando dados completos)
        print("\n>> Otimizando com Gurobi (dados 2021-2024)...")
        start_completo = time.time()
        res_gurobi_completo = modelo_GUROBI.resolver_com_gurobi_setores(
            inputs_completo, lambda_risco, risco_teto,
            warm_start_pesos=None, setores_proibidos=setores_proibidos,
            teto_maximo_ativo=teto_ativo_input, teto_maximo_setor=teto_setor_input,
            max_ativos_carteira=max_ativos_global,
            max_ativos_setor=max_ativos_por_setor
        )
        tempo_completo = time.time() - start_completo
        
        if res_gurobi_completo is None:
            return jsonify({'sucesso': False, 'erro': 'Otimização completa falhou.'}), 400
        
        pesos_completo = res_gurobi_completo['pesos']
        lotes_completo = res_gurobi_completo.get('lotes')
        
        # Métricas completas
        aloc_setor_completo = calcular_alocacao_setorial(nomes_ativos_completo, pesos_completo, valor_investir, precos_map_completo, lotes_completo)
        n_ativos_completo, n_setores_completo = contar_ativos_setores(pesos_completo, aloc_setor_completo)
        
        metricas_completo = {
            'periodo': f"{config.DATA_INICIO_COMPLETO} a {config.DATA_FIM_COMPLETO}",
            'valor_investido': safe_num(valor_investir),
            'retorno_aa': safe_num(res_gurobi_completo['retorno'] * 100),
            'risco_aa': safe_num(res_gurobi_completo['risco'] * 100),
            'score': safe_num(res_gurobi_completo['obj']),
            'tempo': safe_num(tempo_completo),
            'pvp': safe_num(res_gurobi_completo.get('pvp_final')),
            'cvar': safe_num(res_gurobi_completo.get('cvar_final', 0) * 100),
            'qtd_ativos': n_ativos_completo,
            'qtd_setores': n_setores_completo
        }
        
        alocacao_completo = formatar_dados_para_frontend(nomes_ativos_completo, pesos_completo, valor_investir, precos_map_completo, lotes_completo)
        
        # Simula performance da carteira ótima em 2023-2024
        print(f"\n>> Simulando performance da carteira ótima em 2023-2024...")
        performance_otima_teste = preparar_dados.simular_performance_periodo(
            pesos_completo,
            nomes_ativos_completo,
            config.DATA_INICIO_TESTE,
            config.DATA_FIM_TESTE,
            valor_inicial=valor_investir
        )
        
        if performance_otima_teste:
            metricas_otima_teste = {
                'periodo': f"{config.DATA_INICIO_TESTE} a {config.DATA_FIM_TESTE}",
                'retorno_realizado_aa': safe_num(performance_otima_teste['retorno_aa'] * 100),
                'risco_realizado_aa': safe_num(performance_otima_teste['risco_aa'] * 100),
                'valor_final': safe_num(performance_otima_teste['valor_final']),
                'retorno_total': safe_num(performance_otima_teste['retorno_total'] * 100)
            }
        else:
            metricas_otima_teste = None
        

        # Fase 4: Comparação dos resultados
        print(f"\n[FASE 4] Comparando resultados...")
        
        # Diferenças entre carteira de treino vs carteira ótima
        dif_retorno_treino = metricas_treino['retorno_aa'] - metricas_completo['retorno_aa']
        dif_risco_treino = metricas_treino['risco_aa'] - metricas_completo['risco_aa']
        dif_score_treino = metricas_treino['score'] - metricas_completo['score']
        
        # Análise da performance real (Comparando retornos REALIZADOS em 2024)
        if metricas_otima_teste:
            retorno_real_treino = metricas_teste['retorno_realizado_aa']
            retorno_real_otima = metricas_otima_teste['retorno_realizado_aa']
            dif_retorno_real = retorno_real_treino - retorno_real_otima
            retorno_ref = retorno_real_otima
            print(f"[DEBUG COMPARAÇÃO] Realizado Treino ({retorno_real_treino:.2f}%) - Realizado Ótima ({retorno_real_otima:.2f}%) = {dif_retorno_real:.2f}%")
        else:
            dif_retorno_real = metricas_teste['retorno_realizado_aa'] - metricas_completo['retorno_aa']
            retorno_ref = metricas_completo['retorno_aa']
            print(f"[DEBUG COMPARAÇÃO] Fallback: Realizado Treino - Esperado Ótima")
        
        comparacao = {
            'diferenca_retorno_treino': safe_num(dif_retorno_treino),
            'diferenca_risco_treino': safe_num(dif_risco_treino),
            'diferenca_score_treino': safe_num(dif_score_treino),
            'diferenca_retorno_real_vs_otimo': safe_num(dif_retorno_real),
            'mensagem': f"A carteira treinada (2021-2023) teve retorno de {metricas_teste['retorno_realizado_aa']:.2f}% a.a. em 2024, contra {retorno_ref:.2f}% a.a. da carteira ótima simulada no mesmo período."
        }
        
        print(f"\n{'='*80}")
        print(f"RESULTADOS DA ANÁLISE TEMPORAL")
        print(f"{'='*80}")
        print(f"Carteira Treino (2021-2022): Retorno={metricas_treino['retorno_aa']:.2f}% | Risco={metricas_treino['risco_aa']:.2f}%")
        print(f"Performance Real (2023-2024): Retorno={metricas_teste['retorno_realizado_aa']:.2f}% | Risco={metricas_teste['risco_realizado_aa']:.2f}%")
        print(f"Carteira Ótima (2021-2024): Retorno={metricas_completo['retorno_aa']:.2f}% | Risco={metricas_completo['risco_aa']:.2f}%")
        print(f"{'='*80}\n")
        
        # Fase 5: Geração dos gráficos
        print("\n>> Gerando gráficos de alocação...")
        
        # Gráfico da carteira de treino
        path_grafico_treino = os.path.join(STATIC_DIR, 'grafico_temporal_treino.png')
        pesos_treino_series = pd.Series(pesos_treino, index=nomes_ativos_treino)
        plot.plot_pizza_por_ativos(
            serie_pesos=pesos_treino_series,
            risco=res_gurobi_treino['risco'],
            retorno=res_gurobi_treino['retorno'],
            valor_investido=valor_investir,
            nome_arquivo=path_grafico_treino,
            titulo_personalizado="Carteira Treino (2021-2023)"
        )
        
        # Gráfico da carteira ótima
        path_grafico_otima = os.path.join(STATIC_DIR, 'grafico_temporal_otima.png')
        pesos_completo_series = pd.Series(pesos_completo, index=nomes_ativos_completo)
        plot.plot_pizza_por_ativos(
            serie_pesos=pesos_completo_series,
            risco=res_gurobi_completo['risco'],
            retorno=res_gurobi_completo['retorno'],
            valor_investido=valor_investir,
            nome_arquivo=path_grafico_otima,
            titulo_personalizado="Carteira Ótima (2021-2024)"
        )
        
        print("✅ Gráficos gerados com sucesso!")
        
        # Resposta final
        return jsonify({
            'sucesso': True,
            'carteira_2021_2022': {
                'metricas_treino': metricas_treino,
                'alocacao': alocacao_treino,
                'alocacao_setorial': aloc_setor_treino,
                'performance_2023_2024': metricas_teste,
                'evolucao_teste': performance_teste.get('evolucao', {})
            },
            'carteira_2021_2024': {
                'metricas': metricas_completo,
                'alocacao': alocacao_completo,
                'alocacao_setorial': aloc_setor_completo,
                'performance_2023_2024': metricas_otima_teste
            },
            'comparacao': comparacao
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'sucesso': False, 'erro': str(e)}), 500


@app.route('/calcular-fronteira', methods=['POST'])

# Função para calcular a fronteira eficiente
def calcular_fronteira():
    try:
        dados = request.json
        # Parâmetros básicos
        valor_investir = float(dados.get('valor') or 0)
        risco_teto = float(dados.get('risco') or 15) / 100.0
        teto_ativo_input = float(dados.get('teto_ativo') or 30.0) / 100.0
        teto_setor_input = float(dados.get('teto_setor') or 100.0) / 100.0
        setores_proibidos = dados.get('proibidos', [])
        max_ativos_global = int(dados.get('max_ativos') or 15)
        max_ativos_por_setor = int(dados.get('max_ativos_setor') or 4)
        
        # Lambdas para a fronteira
        lambdas_fronteira = [1, 10, 25, 50, 100, 200, 500]
        
        print(f"\n--- [POST /calcular-fronteira] Iniciando cálculo paralelo para lambdas: {lambdas_fronteira} ---")
        
        inputs = None
        with CACHE_LOCK:
            if CACHE_DADOS is not None:
                inputs = CACHE_DADOS.copy()
                inputs['valor_total_investido'] = valor_investir
        
        if inputs is None:
            # Tenta recalcular se não tiver cache
            inputs = preparar_dados.calcular_inputs_otimizacao(valor_investir)
            
        if inputs is None:
            return jsonify({'sucesso': False, 'erro': 'Dados não disponíveis.'}), 500

        # Função auxiliar para calcular um ponto da fronteira
        def calcular_ponto(lam):
            try:
                # 1. Algoritmo Genético
                res_ga = modelo_AG.rodar_otimização(inputs, risco_teto, float(lam), setores_proibidos, 
                                                   teto_maximo_ativo=teto_ativo_input, 
                                                   teto_maximo_setor=teto_setor_input,
                                                   verbose=False)
                
                if not res_ga: return None

                # 2. Gurobi Warm
                res_gu_warm = modelo_GUROBI.resolver_com_gurobi_setores(
                    inputs, float(lam), risco_teto, 
                    warm_start_pesos=res_ga['pesos_finais'], setores_proibidos=setores_proibidos, 
                    teto_maximo_ativo=teto_ativo_input, teto_maximo_setor=teto_setor_input, 
                    max_ativos_carteira=max_ativos_global, max_ativos_setor=max_ativos_por_setor,
                    verbose=False
                )

                # 3. Gurobi Cold
                res_gu_cold = modelo_GUROBI.resolver_com_gurobi_setores(
                    inputs, float(lam), risco_teto, 
                    warm_start_pesos=None, setores_proibidos=setores_proibidos, 
                    teto_maximo_ativo=teto_ativo_input, teto_maximo_setor=teto_setor_input, 
                    max_ativos_carteira=max_ativos_global, max_ativos_setor=max_ativos_por_setor,
                    verbose=False
                )

                return {
                    'lambda': lam,
                    'ga': {'risco': res_ga['risco_final'] * 100, 'retorno': res_ga['retorno_final'] * 100} if res_ga else None,
                    'gu_warm': {'risco': res_gu_warm['risco'] * 100, 'retorno': res_gu_warm['retorno'] * 100} if res_gu_warm else None,
                    'gu_cold': {'risco': res_gu_cold['risco'] * 100, 'retorno': res_gu_cold['retorno'] * 100} if res_gu_cold else None
                }
            except Exception as e:
                print(f"Erro no lambda {lam}: {e}")
                return None

        # Execução Paralela
        resultados = []
        with  concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(calcular_ponto, lam): lam for lam in lambdas_fronteira}
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res: resultados.append(res)
        
        # Organiza dados para o frontend
        resultados.sort(key=lambda x: x['lambda'])
        
        fronteira_ga = [{'x': r['ga']['risco'], 'y': r['ga']['retorno'], 'lambda': r['lambda']} for r in resultados if r['ga']]
        fronteira_gu_warm = [{'x': r['gu_warm']['risco'], 'y': r['gu_warm']['retorno'], 'lambda': r['lambda']} for r in resultados if r['gu_warm']]
        fronteira_gu_cold = [{'x': r['gu_cold']['risco'], 'y': r['gu_cold']['retorno'], 'lambda': r['lambda']} for r in resultados if r['gu_cold']]

        print("--- [POST /calcular-fronteira] Cálculo finalizado. ---")
        return jsonify({
            'sucesso': True,
            'fronteira': {
                'ga': fronteira_ga,
                'gurobi_warm': fronteira_gu_warm,
                'gurobi_cold': fronteira_gu_cold
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'sucesso': False, 'erro': str(e)}), 500


if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print("✅ Servidor rodando! Acesse: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)