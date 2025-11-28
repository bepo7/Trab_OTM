import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os
from bcb import sgs
import warnings
from concurrent.futures import ThreadPoolExecutor

# --- 1. IMPORTAR CONFIGURAÇÕES ---
import config

# --- 2. PARÂMETROS ---
DIAS_UTEIS_ANO = 252
ARQUIVO_CACHE_BENCH = "benchmarks_cache.csv" # Nome do arquivo de cache

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- FUNÇÃO DE EMERGÊNCIA (SIMULAÇÃO) ---
def gerar_dados_sinteticos(data_inicio, data_fim_str):
    """
    Gera dados matemáticos caso TUDO falhe (API e Cache).
    Garante que o usuário veja linhas no gráfico.
    """
    print("⚠️ MODO DE EMERGÊNCIA: Gerando dados sintéticos para Benchmarks...")
    dt_inicio = pd.to_datetime(data_inicio)
    dt_fim = pd.to_datetime(data_fim_str)
    
    # Gera dias úteis
    datas = pd.date_range(start=dt_inicio, end=dt_fim, freq='B')
    n_dias = len(datas)
    
    # 1. Simula CDI (Crescimento exponencial suave ~11% a.a)
    taxa_diaria_cdi = (1 + 0.11) ** (1/252) - 1
    cdi_simulado = np.cumprod(np.repeat(1 + taxa_diaria_cdi, n_dias))
    
    # 2. Simula Ibovespa (Random Walk com leve tendência de alta)
    mu = 0.08 / 252
    sigma = 0.20 / np.sqrt(252)
    retornos_ibov = np.random.normal(mu, sigma, n_dias)
    ibov_simulado = np.cumprod(1 + retornos_ibov)
    
    # 3. Simula S&P 500 (Levemente superior ao IBOV na simulação)
    sp500_simulado = ibov_simulado * 1.05
    
    df = pd.DataFrame({
        'CDI': cdi_simulado / cdi_simulado[0],
        'Ibovespa': ibov_simulado / ibov_simulado[0],
        'S&P500 (BRL)': sp500_simulado / sp500_simulado[0]
    }, index=datas)
    
    return df

# --- FUNÇÃO DE DOWNLOAD ROBUSTA ---
def baixar_benchmarks(data_inicio, data_fim_str):
    print(f"Obtendo benchmarks (Inicio: {data_inicio})...")
    
    df_api = None
    sucesso_api = False
    
    # --- TENTATIVA 1: APIS ONLINE ---
    try:
        dt_fim_ajustada = pd.to_datetime(data_fim_str) + datetime.timedelta(days=1)

        # 1.1 CDI (Banco Central)
        try:
            cdi_diario = sgs.get({'CDI': 12}, start=data_inicio, end=data_fim_str)
            if not cdi_diario.empty:
                cdi_fator = (1 + cdi_diario / 100)
                cdi_acumulado = cdi_fator.cumprod()
                cdi_acumulado = cdi_acumulado / cdi_acumulado.iloc[0]
                cdi_acumulado.columns = ['CDI']
            else:
                cdi_acumulado = pd.DataFrame()
        except:
            cdi_acumulado = pd.DataFrame()

        # 1.2 Ibovespa e S&P500 (Via ETF IVVB11)
        try:
            dados_bench = yf.download(['^BVSP', 'IVVB11.SA'], start=data_inicio, end=dt_fim_ajustada, progress=False, auto_adjust=True)
            
            if 'Close' in dados_bench.columns:
                precos_bench = dados_bench['Close']
            else:
                precos_bench = dados_bench
                
            # Limpeza e Normalização
            bench_normalizado = precos_bench.ffill().bfill() # bfill evita buracos no inicio
            if not bench_normalizado.empty:
                bench_normalizado = bench_normalizado / bench_normalizado.iloc[0]
                bench_normalizado = bench_normalizado.rename(columns={'^BVSP': 'Ibovespa', 'IVVB11.SA': 'S&P500 (BRL)'})
            else:
                bench_normalizado = pd.DataFrame()
                
        except:
            bench_normalizado = pd.DataFrame()

        # Junta tudo
        if not cdi_acumulado.empty and not bench_normalizado.empty:
            df_api = pd.concat([cdi_acumulado, bench_normalizado], axis=1).ffill().bfill()
            # Garante base 1.0
            df_api = df_api / df_api.iloc[0]
            sucesso_api = True
            print("✅ Benchmarks baixados com sucesso.")
    
    except Exception as e:
        print(f"Erro APIs: {e}")

    # --- TENTATIVA 2: CACHE LOCAL ---
    if sucesso_api and df_api is not None:
        try: df_api.to_csv(ARQUIVO_CACHE_BENCH)
        except: pass
        return df_api
            
    if os.path.exists(ARQUIVO_CACHE_BENCH):
        print("⚠️ Usando Cache Local...")
        try:
            df_cache = pd.read_csv(ARQUIVO_CACHE_BENCH, index_col=0, parse_dates=True)
            df_cache = df_cache[df_cache.index >= data_inicio]
            if not df_cache.empty:
                return df_cache / df_cache.iloc[0]
        except: pass
        
    # --- TENTATIVA 3: SIMULAÇÃO ---
    return gerar_dados_sinteticos(data_inicio, data_fim_str)

# --- OUTRAS FUNÇÕES AUXILIARES (PADRÃO) ---

def baixar_dados_com_volume(lista_de_tickers, data_inicio, data_fim):
    if not lista_de_tickers: return None, None
    data_fim_ajustada = data_fim + datetime.timedelta(days=1)
    try:
        dados = yf.download(lista_de_tickers, start=data_inicio, end=data_fim_ajustada, progress=False, auto_adjust=True)
        if dados.empty: return None, None
        
        if 'Close' in dados.columns: precos = dados['Close']
        else: precos = dados
        if 'Volume' in dados.columns: volumes = dados['Volume']
        else: volumes = pd.DataFrame(np.nan, index=precos.index, columns=precos.columns)

        if len(lista_de_tickers) == 1:
            if isinstance(precos, pd.Series): precos = precos.to_frame(lista_de_tickers[0])
            if isinstance(volumes, pd.Series): volumes = volumes.to_frame(lista_de_tickers[0])

        precos = precos.dropna(axis=1, how='all').ffill().bfill()
        volumes = volumes.dropna(axis=1, how='all').fillna(0)
        ativos_comuns = precos.columns.intersection(volumes.columns)
        return precos[ativos_comuns], volumes[ativos_comuns]
    except: return None, None

def simular_evolucao_diaria(retornos_hist, pesos, valor_inicial=100):
    pesos_series = pd.Series(pesos, index=retornos_hist.columns)
    retorno_diario = (retornos_hist * pesos_series).sum(axis=1).fillna(0)
    acumulado = (1 + retorno_diario).cumprod() * valor_inicial
    valores = acumulado.values.tolist()
    datas = [d.strftime('%Y-%m-%d') for d in acumulado.index]
    return datas, valores

def pegar_pvp_individual(ticker):
    try:
        t = yf.Ticker(ticker)
        val = t.info.get('priceToBook')
        return ticker, (float(val) if val is not None and float(val) > 0 else 20.0)
    except: return ticker, np.nan

def obter_pvp_ativos_otimizado(lista_tickers):
    pvp_data = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        for ticker, valor in executor.map(pegar_pvp_individual, lista_tickers):
            pvp_data[ticker] = valor
    return pd.Series(pvp_data).fillna(1.0)

def calcular_cvar_95(retornos):
    cvar_dict = {}
    for ativo in retornos.columns:
        rets = retornos[ativo].values
        if len(rets) == 0: cvar_dict[ativo] = 0.05
        else:
            rets_sorted = np.sort(rets)
            cutoff = max(1, int(len(rets) * 0.05))
            cvar_dict[ativo] = abs(rets_sorted[:cutoff].mean())
    return pd.Series(cvar_dict)

def limpar_dados(retornos_medios, matriz_cov, volumes_medios):
    mask = ~np.isfinite(retornos_medios)
    if mask.any(): retornos_medios = retornos_medios.drop(retornos_medios[mask].index)
    diag = np.diag(matriz_cov)
    mask_var = ~np.isfinite(diag)
    if mask_var.any(): matriz_cov = matriz_cov.drop(index=matriz_cov.index[mask_var], columns=matriz_cov.columns[mask_var])
    volumes_medios = volumes_medios.reindex(retornos_medios.index).fillna(0)
    return retornos_medios, matriz_cov, volumes_medios

# --- FUNÇÃO PRINCIPAL (ATUALIZADA) ---
def calcular_inputs_otimizacao(valor_total_investido):
    lista_ativos = config.UNIVERSO_COMPLETO
    if not lista_ativos: return None

    data_fim = datetime.date.today()
    data_inicio = data_fim - datetime.timedelta(days=config.ANOS_DE_DADOS * 365.25)
    
    precos, volumes = baixar_dados_com_volume(lista_ativos, data_inicio, data_fim)
    if precos is None or precos.empty: return None

    retornos_diarios = precos.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if retornos_diarios.empty: return None
    
    retornos_medios = retornos_diarios.mean() * DIAS_UTEIS_ANO
    matriz_cov = retornos_diarios.cov() * DIAS_UTEIS_ANO
    
    preco_medio = precos.tail(126).mean()
    vol_qtd = volumes.tail(126).mean()
    volume_financeiro = (vol_qtd * preco_medio).fillna(0)

    retornos_medios, matriz_cov, volume_financeiro = limpar_dados(retornos_medios, matriz_cov, volume_financeiro)
    
    ativos_validos = list(retornos_medios.index)
    ret_validos = retornos_diarios[ativos_validos] 
    
    vetor_cvar = calcular_cvar_95(ret_validos)
    vetor_pvp = obter_pvp_ativos_otimizado(ativos_validos)
    
    # 6. BAIXAR E ALINHAR BENCHMARKS
    inicio_real = ret_validos.index[0].strftime('%Y-%m-%d')
    df_benchmarks = baixar_benchmarks(inicio_real, data_fim.strftime('%Y-%m-%d'))
    
    # ALINHAMENTO CRITICO: Usa reindex do PANDAS para forçar as mesmas datas
    # O .bfill() preenche buracos no inicio para evitar NaN (que vira zero no grafico)
    df_benchmarks = df_benchmarks.reindex(ret_validos.index).ffill().bfill()
    
    # Garante normalização base 1.0
    if not df_benchmarks.empty:
        df_benchmarks = df_benchmarks / df_benchmarks.iloc[0]

    # Garante que as colunas existam (mesmo que duplicadas) para não quebrar o código
    if 'CDI' not in df_benchmarks.columns: df_benchmarks['CDI'] = 1.0
    if 'Ibovespa' not in df_benchmarks.columns: df_benchmarks['Ibovespa'] = 1.0
    if 'S&P500 (BRL)' not in df_benchmarks.columns: df_benchmarks['S&P500 (BRL)'] = df_benchmarks['Ibovespa']

    print("\n--- Inputs Prontos ---")
    
    return {
        'valor_total_investido': valor_total_investido,
        'retornos_medios': retornos_medios,
        'matriz_cov': matriz_cov,
        'vetor_pvp': vetor_pvp.reindex(ativos_validos).fillna(1.0),
        'vetor_cvar': vetor_cvar.reindex(ativos_validos).fillna(0.05),
        'volume_medio': volume_financeiro.reindex(ativos_validos).fillna(0),
        'nomes_dos_ativos': ativos_validos,
        'n_ativos': len(ativos_validos),
        'retornos_diarios_historicos': ret_validos, 
        'df_benchmarks': df_benchmarks
    }