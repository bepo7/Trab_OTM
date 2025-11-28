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
BENCHMARK_MERCADO = '^BVSP'
CDI_CODIGO_BCB = 4389
TAXA_LIVRE_RISCO_FALLBACK = 0.15 
DIAS_UTEIS_ANO = 252
ARQUIVO_CACHE_BENCH = "benchmarks_cache.csv" # Nome do arquivo de cache

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- FUNÇÕES AUXILIARES DE DADOS ---

def gerar_dados_sinteticos(data_inicio, data_fim_str):
    """
    Gera dados matemáticos caso TUDO falhe (API e Cache).
    Garante que o usuário veja linhas no gráfico.
    """
    print("⚠️ MODO DE EMERGÊNCIA: Gerando dados sintéticos para Benchmarks...")
    dt_inicio = pd.to_datetime(data_inicio)
    dt_fim = pd.to_datetime(data_fim_str)
    
    # Gera dias úteis (aproximado)
    datas = pd.date_range(start=dt_inicio, end=dt_fim, freq='B')
    n_dias = len(datas)
    
    # 1. Simula CDI (Crescimento exponencial suave ~11% a.a)
    taxa_diaria_cdi = (1 + 0.11) ** (1/252) - 1
    cdi_simulado = np.cumprod(np.repeat(1 + taxa_diaria_cdi, n_dias))
    # Normaliza para começar em 1.0
    cdi_simulado = cdi_simulado / cdi_simulado[0]
    
    # 2. Simula Ibovespa (Random Walk com leve tendência de alta ~8% a.a e vol 20%)
    mu = 0.08 / 252
    sigma = 0.20 / np.sqrt(252)
    retornos = np.random.normal(mu, sigma, n_dias)
    ibov_simulado = np.cumprod(1 + retornos)
    ibov_simulado = ibov_simulado / ibov_simulado[0]
    
    df = pd.DataFrame({
        'CDI': cdi_simulado,
        'Ibovespa': ibov_simulado,
        'S&P500 (BRL)': ibov_simulado * 1.05 # S&P levemente melhor simulado
    }, index=datas)
    
    return df

def baixar_benchmarks(data_inicio, data_fim_str):
    """
    Lógica em 3 camadas:
    1. Tenta baixar da API (BCB e Yahoo). Se der certo, salva Cache.
    2. Se falhar, tenta ler do Cache local.
    3. Se não tiver Cache, gera dados sintéticos.
    """
    print(f"Obtendo benchmarks (Inicio: {data_inicio})...")
    
    df_api = None
    sucesso_api = False
    
    # --- TENTATIVA 1: APIS ONLINE ---
    try:
        # Ajuste de datas
        dt_fim = datetime.datetime.strptime(data_fim_str, '%Y-%m-%d').date()
        dt_fim_ajustada = dt_fim + datetime.timedelta(days=1)

        # 1.1 CDI (Banco Central)
        try:
            cdi_diario = sgs.get({'CDI': 12}, start=data_inicio, end=data_fim_str)
            if cdi_diario.empty: raise ValueError("BCB retornou vazio")
            
            cdi_fator = (1 + cdi_diario / 100)
            cdi_acumulado = cdi_fator.cumprod()
            cdi_acumulado = cdi_acumulado / cdi_acumulado.iloc[0]
            cdi_acumulado.columns = ['CDI']
        except Exception as e:
            print(f"Falha API BCB: {e}")
            cdi_acumulado = pd.DataFrame()

        # 1.2 Ibovespa e S&P (Yahoo)
        try:
            tickers_bench = ['^BVSP', 'IVVB11.SA'] 
            dados_bench = yf.download(tickers_bench, start=data_inicio, end=dt_fim_ajustada, progress=False, auto_adjust=True)
            
            if 'Close' in dados_bench.columns:
                precos_bench = dados_bench['Close']
            else:
                precos_bench = dados_bench
                
            bench_normalizado = precos_bench.ffill().bfill()
            if not bench_normalizado.empty:
                bench_normalizado = bench_normalizado / bench_normalizado.iloc[0]
                bench_normalizado = bench_normalizado.rename(columns={'^BVSP': 'Ibovespa', 'IVVB11.SA': 'S&P500 (BRL)'})
            else:
                bench_normalizado = pd.DataFrame()
                
        except Exception as e:
            print(f"Falha API Yahoo: {e}")
            bench_normalizado = pd.DataFrame()

        # Junta API
        if not cdi_acumulado.empty and not bench_normalizado.empty:
            df_api = pd.concat([cdi_acumulado, bench_normalizado], axis=1).ffill()
            # Remove linhas que ficaram vazias no começo e garante normalização
            df_api = df_api.dropna()
            if not df_api.empty:
                # Normaliza base 1.0 novamente para garantir alinhamento
                df_api = df_api / df_api.iloc[0]
                sucesso_api = True
                print("✅ Benchmarks baixados com sucesso.")
    
    except Exception as e:
        print(f"Erro geral APIs: {e}")

    # --- SALVAR OU RECUPERAR CACHE ---
    
    if sucesso_api and df_api is not None:
        # Salva para uso futuro
        try:
            df_api.to_csv(ARQUIVO_CACHE_BENCH)
            return df_api
        except:
            return df_api # Retorna mesmo se falhar ao salvar
            
    else:
        # --- TENTATIVA 2: LER CACHE ---
        print("⚠️ Falha nas APIs. Tentando carregar Cache local...")
        if os.path.exists(ARQUIVO_CACHE_BENCH):
            try:
                df_cache = pd.read_csv(ARQUIVO_CACHE_BENCH, index_col=0, parse_dates=True)
                # Filtra apenas datas relevantes (corte temporal)
                df_cache = df_cache[df_cache.index >= data_inicio]
                
                if not df_cache.empty:
                    # Renormaliza para a nova data de início ser 1.0
                    df_cache = df_cache / df_cache.iloc[0]
                    print("✅ Dados recuperados do Cache.")
                    return df_cache
            except Exception as e:
                print(f"Erro ao ler cache: {e}")
        
        # --- TENTATIVA 3: SIMULAÇÃO (FALLBACK FINAL) ---
        return gerar_dados_sinteticos(data_inicio, data_fim_str)

def baixar_dados_com_volume(lista_de_tickers, data_inicio, data_fim):
    """ Baixa Preços e Volume com data final ajustada """
    if not lista_de_tickers: return None, None
    
    data_fim_ajustada = data_fim + datetime.timedelta(days=1)
    
    print(f"Baixando dados para {len(lista_de_tickers)} ativos...")
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

    except Exception as e:
        print(f"Erro download: {e}")
        return None, None

def simular_evolucao_diaria(retornos_hist, pesos, valor_inicial=100):
    """ Calcula evolução diária para o gráfico JS """
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
        if val is None: return ticker, np.nan
        val_float = float(val)
        if val_float <= 0: return ticker, 20.0 
        return ticker, val_float
    except: return ticker, np.nan

def obter_pvp_ativos_otimizado(lista_tickers):
    print(f"\nBaixando P/VP (Paralelo)...")
    pvp_data = {}
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        resultados = executor.map(pegar_pvp_individual, lista_tickers)
        for ticker, valor in resultados:
            pvp_data[ticker] = valor
            
    serie = pd.Series(pvp_data)
    media = serie.mean()
    if np.isnan(media): media = 1.0
    
    return serie.fillna(media)

def calcular_cvar_95(retornos):
    cvar_dict = {}
    for ativo in retornos.columns:
        rets = retornos[ativo].values
        if len(rets) == 0:
            cvar_dict[ativo] = 0.05
            continue
            
        rets_sorted = np.sort(rets)
        cutoff = int(len(rets) * 0.05)
        if cutoff < 1: cutoff = 1
        
        cvar_dict[ativo] = abs(rets_sorted[:cutoff].mean())
        
    return pd.Series(cvar_dict)

def limpar_dados(retornos_medios, matriz_cov, volumes_medios):
    ativos_removidos = []
    
    mask = ~np.isfinite(retornos_medios)
    if mask.any():
        bad = retornos_medios[mask].index.tolist()
        ativos_removidos.extend(bad)

    diag = np.diag(matriz_cov)
    mask_var = ~np.isfinite(diag)
    if mask_var.any():
        bad = matriz_cov.index[np.where(mask_var)[0]].tolist()
        ativos_removidos.extend(bad)
        
    ativos_removidos = list(set(ativos_removidos))
    
    if ativos_removidos:
        retornos_medios = retornos_medios.drop(ativos_removidos, errors='ignore')
        matriz_cov = matriz_cov.drop(index=ativos_removidos, columns=ativos_removidos, errors='ignore')
        volumes_medios = volumes_medios.drop(ativos_removidos, errors='ignore')
        
    return retornos_medios, matriz_cov, volumes_medios

# --- FUNÇÃO PRINCIPAL ---

def calcular_inputs_otimizacao(valor_total_investido):
    lista_ativos = config.UNIVERSO_COMPLETO
    if not lista_ativos: return None

    # Datas
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

    retornos_medios, matriz_cov, volume_financeiro = limpar_dados(
        retornos_medios, matriz_cov, volume_financeiro
    )
    
    ativos_validos = list(retornos_medios.index)
    ret_validos = retornos_diarios[ativos_validos] 
    
    vetor_cvar = calcular_cvar_95(ret_validos)
    vetor_pvp = obter_pvp_ativos_otimizado(ativos_validos)
    
    vetor_cvar = vetor_cvar.reindex(ativos_validos).fillna(0.05)
    vetor_pvp = vetor_pvp.reindex(ativos_validos).fillna(1.0)
    volume_financeiro = volume_financeiro.reindex(ativos_validos).fillna(0)
    
    # 6. BAIXAR BENCHMARKS (COM LOGICA ROBUSTA)
    inicio_real = ret_validos.index[0].strftime('%Y-%m-%d')
    df_benchmarks = baixar_benchmarks(inicio_real, data_fim.strftime('%Y-%m-%d'))
    
    # Alinhamento final: Intersecção de datas entre Carteira e Benchmark
    idx_comum = ret_validos.index.intersection(df_benchmarks.index)
    
    # Filtra e preenche
    if len(idx_comum) > 0:
        df_benchmarks = df_benchmarks.loc[idx_comum]
        # Re-normaliza para 1.0 no dia comum inicial
        df_benchmarks = df_benchmarks / df_benchmarks.iloc[0]
        # Reindexa para bater com retornos (preenche buracos se houver)
        df_benchmarks = df_benchmarks.reindex(ret_validos.index).ffill()
    else:
        # Se as datas não baterem (ex: simulado vs real desalinhado), 
        # força o benchmark a usar o índice da carteira (copiando dados sinteticos)
        df_benchmarks = df_benchmarks.iloc[:len(ret_validos)]
        df_benchmarks.index = ret_validos.index

    print("\n--- Inputs Prontos ---")
    
    return {
        'valor_total_investido': valor_total_investido,
        'retornos_medios': retornos_medios,
        'matriz_cov': matriz_cov,
        'vetor_pvp': vetor_pvp,
        'vetor_cvar': vetor_cvar,
        'volume_medio': volume_financeiro,
        'nomes_dos_ativos': ativos_validos,
        'n_ativos': len(ativos_validos),
        'retornos_diarios_historicos': ret_validos, 
        'df_benchmarks': df_benchmarks
    }