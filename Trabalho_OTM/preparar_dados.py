import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os
from bcb import sgs
import warnings
from concurrent.futures import ThreadPoolExecutor


import config


DIAS_UTEIS_ANO = 252
ARQUIVO_CACHE_BENCH = "Trabalho_OTM/valores_benchmarks.csv" # Nome do arquivo de cache

warnings.simplefilter(action='ignore', category=FutureWarning)

# Download dos benchmarks
def baixar_benchmarks(data_inicio, data_fim_str):
    print(f"Obtendo benchmarks (Inicio: {data_inicio})...")
    
    df_api = None
    sucesso_api = False
    
    # Tentativa 1: BCB + Yahoo Finance
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

        # 1.2 Ibovespa e S&P500 (Yahoo Finance)
        try:
            dados_bench = yf.download(['^BVSP', 'IVVB11.SA'], start=data_inicio, end=dt_fim_ajustada, progress=False, auto_adjust=True)
            
            if 'Close' in dados_bench.columns:
                precos_bench = dados_bench['Close']
            else:
                precos_bench = dados_bench
                
            # Trata caso só tenha um ativo
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
            if hasattr(cdi_acumulado.index, 'tz') and cdi_acumulado.index.tz is not None:
                cdi_acumulado.index = cdi_acumulado.index.tz_localize(None)
            if hasattr(bench_normalizado.index, 'tz') and bench_normalizado.index.tz is not None:
                bench_normalizado.index = bench_normalizado.index.tz_localize(None)
            
            df_api = pd.concat([cdi_acumulado, bench_normalizado], axis=1).ffill().bfill()
            df_api = df_api / df_api.iloc[0]
            sucesso_api = True
            print("✅ Benchmarks baixados com sucesso.")
    
    except Exception as e:
        print(f"Erro APIs: {e}")

    # Tentativa 2: Verifica Cache Local
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
        
    # Tentativa 3: Gera dados sintéticos
    return gerar_dados_sinteticos(data_inicio, data_fim_str)

# Caso de erro nos downloads, gera dados sintéticos
def gerar_dados_sinteticos(data_inicio, data_fim_str):
    print("⚠️ MODO DE EMERGÊNCIA: Gerando dados sintéticos para Benchmarks...")
    dt_inicio = pd.to_datetime(data_inicio)
    dt_fim = pd.to_datetime(data_fim_str)
    
    # Gera dias úteis
    datas = pd.date_range(start=dt_inicio, end=dt_fim, freq='B')
    n_dias = len(datas)
    
    # 1. Simula CDI 
    taxa_diaria_cdi = (1 + 0.11) ** (1/252) - 1
    cdi_simulado = np.cumprod(np.repeat(1 + taxa_diaria_cdi, n_dias))
    
    # 2. Simula Ibovespa 
    mu = 0.08 / 252
    sigma = 0.20 / np.sqrt(252)
    retornos_ibov = np.random.normal(mu, sigma, n_dias)
    ibov_simulado = np.cumprod(1 + retornos_ibov)
    
    # 3. Simula S&P 500 
    sp500_simulado = ibov_simulado * 1.05
    
    df = pd.DataFrame({
        'CDI': cdi_simulado / cdi_simulado[0],
        'Ibovespa': ibov_simulado / ibov_simulado[0],
        'S&P500 (BRL)': sp500_simulado / sp500_simulado[0]
    }, index=datas)
    
    return df

# Baixa dados com preços e volumes para ser usado na liquidez
def baixar_dados_com_volume(lista_de_tickers, data_inicio, data_fim):
    if not lista_de_tickers: return None, None
    data_fim_ajustada = data_fim + datetime.timedelta(days=1)
    try:
        dados = yf.download(lista_de_tickers, start=data_inicio, end=data_fim_ajustada, progress=False, auto_adjust=True)
        if dados.empty: return None, None
        
        # Extrai preços e volumes
        if 'Close' in dados.columns: precos = dados['Close']
        else: precos = dados
        if 'Volume' in dados.columns: volumes = dados['Volume']
        else: volumes = pd.DataFrame(np.nan, index=precos.index, columns=precos.columns)

        # Trata caso só tenha um ativo
        if len(lista_de_tickers) == 1:
            if isinstance(precos, pd.Series): precos = precos.to_frame(lista_de_tickers[0])
            if isinstance(volumes, pd.Series): volumes = volumes.to_frame(lista_de_tickers[0])
        
        precos = precos.dropna(axis=1, how='all')
        precos = precos.ffill().bfill()
        
        if hasattr(precos.index, 'tz') and precos.index.tz is not None:
            precos.index = precos.index.tz_localize(None)
        
        volumes = volumes.dropna(axis=1, how='all').fillna(0)

        if hasattr(volumes.index, 'tz') and volumes.index.tz is not None:
            volumes.index = volumes.index.tz_localize(None)
        
        ativos_comuns = precos.columns.intersection(volumes.columns)

        if len(ativos_comuns) == 0:
            print("⚠️ Nenhum ativo válido após processamento")
            return None, None

        print(f"✅ Dados baixados. Ativos válidos: {len(ativos_comuns)}")
        return precos[ativos_comuns], volumes[ativos_comuns]
    
    except TypeError as e:
        if "Cannot join tz-naive with tz-aware" in str(e):
            print(f"⚠️ Erro de timezone no download em lote. Tentando download individual...")
            # Caso de erro de timezone, tenta baixar individualmente
            all_precos = []
            all_volumes = []
            for ticker in lista_de_tickers:
                try:
                    # Baixa individualmente os dados
                    dados_individual = yf.download([ticker], start=data_inicio, end=data_fim_ajustada, progress=False, auto_adjust=True)
                    if not dados_individual.empty:
                        if 'Close' in dados_individual.columns:
                            preco = dados_individual['Close']
                        else:
                            preco = dados_individual
                        if 'Volume' in dados_individual.columns:
                            volume = dados_individual['Volume']
                        else:
                            volume = pd.Series(np.nan, index=preco.index)
                        
                        if hasattr(preco.index, 'tz') and preco.index.tz is not None:
                            preco.index = preco.index.tz_localize(None)
                        if hasattr(volume.index, 'tz') and volume.index.tz is not None:
                            volume.index = volume.index.tz_localize(None)
                        
                        preco.name = ticker
                        volume.name = ticker
                        all_precos.append(preco)
                        all_volumes.append(volume)
                except:
                    continue
            
            # Consolida resultados
            if all_precos:
                precos = pd.concat(all_precos, axis=1).ffill().bfill()
                volumes = pd.concat(all_volumes, axis=1).fillna(0)
                print(f"✅ Dados baixados individualmente. Ativos válidos: {len(precos.columns)}")
                return precos, volumes
            else:
                print("❌ Nenhum ativo baixado com sucesso")
                return None, None
        else:
            import traceback
            print(f"❌ Erro de tipo no download: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None, None
    except KeyError as e:
        print(f"❌ Erro de chave no download: {e}")
        return None, None
    except ValueError as e:
        print(f"❌ Erro de valor no download: {e}")
        return None, None
    except Exception as e: 
        import traceback
        print(f"❌ Erro inesperado no download: {type(e).__name__}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None, None

# Simula evolução diária da carteira para o gráfico do site
def simular_evolucao_diaria(retornos_hist, pesos, valor_inicial=100):
    # Calcula retorno diário da carteira e evolução acumulada
    pesos_series = pd.Series(pesos, index=retornos_hist.columns)
    retorno_diario = (retornos_hist * pesos_series).sum(axis=1).fillna(0)
    acumulado = (1 + retorno_diario).cumprod() * valor_inicial
    valores = acumulado.values.tolist()
    datas = [d.strftime('%Y-%m-%d') for d in acumulado.index]
    return datas, valores

# Simula performance da carteira em um período específico
def simular_performance_periodo(pesos_carteira, nomes_ativos, data_inicio, data_fim, valor_inicial=100000):
    if isinstance(data_inicio, str):
        data_inicio = datetime.datetime.strptime(data_inicio, '%Y-%m-%d').date()
    if isinstance(data_fim, str):
        data_fim = datetime.datetime.strptime(data_fim, '%Y-%m-%d').date()
    
    print(f"\n--- Simulando performance de {data_inicio} a {data_fim} ---")
    
    # Baixa dados do período
    data_fim_ajustada = data_fim + datetime.timedelta(days=1)
    try:
        dados = yf.download(nomes_ativos, start=data_inicio, end=data_fim_ajustada, progress=False, auto_adjust=True)
        if dados.empty:
            print("⚠️ Sem dados para o período de simulação")
            return None
        
        if 'Close' in dados.columns:
            precos = dados['Close']
        else:
            precos = dados
        
        if len(nomes_ativos) == 1:
            if isinstance(precos, pd.Series):
                precos = precos.to_frame(nomes_ativos[0])
        
        precos = precos.dropna(axis=1, how='all').ffill().bfill()
        
        # Filtra explicitamente para garantir que apenas dados do período sejam usados
        precos = precos.loc[data_inicio:data_fim]
        
        if precos.empty:
            print("⚠️ Sem dados após filtro de datas")
            return None
        
        print(f"[DEBUG] Preços filtrados: {len(precos)} dias de {precos.index[0].strftime('%Y-%m-%d')} a {precos.index[-1].strftime('%Y-%m-%d')}")
        
        # Calcula retornos diários
        retornos_diarios = precos.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        
        print(f"[DEBUG] Retornos calculados: {len(retornos_diarios)} dias úteis")
        
        if retornos_diarios.empty:
            print("⚠️ Sem retornos válidos para simulação")
            return None
        
        # Alinha pesos com os ativos disponíveis
        if isinstance(pesos_carteira, pd.Series):
            pesos_alinhados = pesos_carteira.reindex(retornos_diarios.columns).fillna(0.0)
        else:
            dict_pesos = dict(zip(nomes_ativos, pesos_carteira))
            pesos_alinhados = pd.Series([dict_pesos.get(col, 0.0) for col in retornos_diarios.columns], 
                                        index=retornos_diarios.columns)
        
        # Normaliza pesos (caso alguns ativos não existam no período)
        soma_pesos = pesos_alinhados.sum()
        valor_inicial_efetivo = valor_inicial  # Valor que realmente será investido
        
        if soma_pesos > 0:
            pesos_alinhados = pesos_alinhados / soma_pesos
        else:
            print("⚠️ Nenhum ativo disponível no período de simulação")
            return None
        
        # Calcula retorno diário da carteira
        retorno_carteira_diario = (retornos_diarios * pesos_alinhados).sum(axis=1)
        
        # Calcula retorno total acumulado
        retorno_acumulado = (1 + retorno_carteira_diario).prod() - 1
        
        # Cálculo do valor final da carteira
        valor_final = valor_inicial_efetivo * (1 + retorno_acumulado)
        retorno_total = retorno_acumulado
        
        # Cálculo de X (anos) para anualização correta
        ano_inicio = pd.to_datetime(data_inicio).year
        ano_fim = pd.to_datetime(data_fim).year
        X = ano_fim - ano_inicio
        if X == 0: X = 1  # Evita divisão por zero se for mesmo ano
        
        # Métricas anualizadas usando X (anos calendário) em vez de dias úteis
        dias_uteis = len(retorno_carteira_diario)
        anos = float(X) # Usa a diferença de anos calendário
        
        # Debug: mostra período real usado
        data_inicio_real = retorno_carteira_diario.index[0].strftime('%Y-%m-%d')
        data_fim_real = retorno_carteira_diario.index[-1].strftime('%Y-%m-%d')
        print(f"[DEBUG SIMULAÇÃO] Período: {data_inicio_real} a {data_fim_real} ({dias_uteis} dias úteis)")
        print(f"[DEBUG SIMULAÇÃO] X (Anos Calendário): {X} | Retorno Total: {retorno_total*100:.2f}%")
        print(f"[DEBUG SIMULAÇÃO] Valor Inicial: R$ {valor_inicial_efetivo:.2f} | Valor Final: R$ {valor_final:.2f}")
        
        if anos > 0:
            retorno_aa = (1 + retorno_total) ** (1 / anos) - 1
        else:
            retorno_aa = 0.0
        
        risco_aa = retorno_carteira_diario.std() * np.sqrt(DIAS_UTEIS_ANO)
        
        
        # Evolução para gráfico (recalcula para visualização)
        evolucao = (1 + retorno_carteira_diario).cumprod() * valor_inicial_efetivo
        datas_evolucao = [d.strftime('%Y-%m-%d') for d in evolucao.index]
        valores_evolucao = evolucao.values.tolist()
        
        print(f"✅ Simulação concluída: Retorno={retorno_aa*100:.2f}% a.a., Risco={risco_aa*100:.2f}% a.a.")
        
        return {
            'retorno_aa': retorno_aa,
            'risco_aa': risco_aa,
            'valor_final': valor_final,
            'retorno_total': retorno_total,
            'evolucao': {'datas': datas_evolucao, 'valores': valores_evolucao},
            'periodo': {'inicio': data_inicio.strftime('%Y-%m-%d'), 'fim': data_fim.strftime('%Y-%m-%d')}
        }
        
    except Exception as e:
        print(f"❌ Erro na simulação: {e}")
        return None

# Pega P/VP individual do ativo
def pegar_pvp_individual(ticker):
    try:
        t = yf.Ticker(ticker)
        val = t.info.get('priceToBook')
        return ticker, (float(val) if val is not None and float(val) > 0 else 2.0)
    except: return ticker, np.nan

# Obtém P/VP para lista de ativos, com cache local
def obter_pvp_ativos_otimizado(lista_tickers):
  
    cache_file = 'Trabalho_OTM/valores_pvp.csv'
    cache_max_age_hours = 24
    
    # Tenta carregar cache existente
    pvp_data = {}
    cache_valido = False
    
    if os.path.exists(cache_file):
        try:
            cache_df = pd.read_csv(cache_file, index_col=0)
            # Verifica idade do cache
            if 'timestamp' in cache_df.columns:
                timestamp = pd.to_datetime(cache_df['timestamp'].iloc[0])
                idade_horas = (datetime.datetime.now() - timestamp).total_seconds() / 3600
                if idade_horas < cache_max_age_hours:
                    cache_valido = True
                    pvp_data = cache_df['pvp'].to_dict()
                    print(f"✅ Cache P/VP carregado ({len(pvp_data)} ativos, {idade_horas:.1f}h atrás)")
        except Exception as e:
            print(f"⚠️ Erro ao carregar cache P/VP: {e}")
    
    # Identifica tickers que precisam ser baixados
    tickers_faltantes = [t for t in lista_tickers if t not in pvp_data]
    
    if tickers_faltantes:
        print(f"📥 Baixando P/VP para {len(tickers_faltantes)} ativos...")
        with ThreadPoolExecutor(max_workers=20) as executor:
            for ticker, valor in executor.map(pegar_pvp_individual, tickers_faltantes):
                pvp_data[ticker] = valor
        
        # Salva cache atualizado
        try:
            cache_df = pd.DataFrame({
                'pvp': pvp_data,
                'timestamp': datetime.datetime.now()
            })
            cache_df.to_csv(cache_file)
            print(f"💾 Cache P/VP salvo em '{cache_file}'")
        except Exception as e:
            print(f"⚠️ Erro ao salvar cache P/VP: {e}")
    
    return pd.Series(pvp_data).reindex(lista_tickers).fillna(1.0)

# Calcula CVaR 95% para cada ativo
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

# Limpa dados removendo NaNs e ajustando matrizes
def limpar_dados(retornos_medios, matriz_cov, volumes_medios):
    mask = ~np.isfinite(retornos_medios)
    if mask.any(): retornos_medios = retornos_medios.drop(retornos_medios[mask].index)
    diag = np.diag(matriz_cov)
    mask_var = ~np.isfinite(diag)
    if mask_var.any(): matriz_cov = matriz_cov.drop(index=matriz_cov.index[mask_var], columns=matriz_cov.columns[mask_var])
    volumes_medios = volumes_medios.reindex(retornos_medios.index).fillna(0)
    return retornos_medios, matriz_cov, volumes_medios

# Função principal para calcular inputs de otimização para um período específico
def calcular_inputs_otimizacao_periodo(valor_total_investido, data_inicio, data_fim):
    
    lista_ativos = config.UNIVERSO_COMPLETO
    if not lista_ativos: return None

    # Converte strings para datetime.date se necessário
    if isinstance(data_inicio, str):
        data_inicio = datetime.datetime.strptime(data_inicio, '%Y-%m-%d').date()
    if isinstance(data_fim, str):
        data_fim = datetime.datetime.strptime(data_fim, '%Y-%m-%d').date()
    
    print(f"\n--- Baixando dados para período: {data_inicio} a {data_fim} ---")
    
    # Baixa preços e volumes
    precos, volumes = baixar_dados_com_volume(lista_ativos, data_inicio, data_fim)
    if precos is None or precos.empty: return None

    # Calcula retornos diários
    retornos_diarios = precos.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if retornos_diarios.empty: return None
    
    # Calcula retornos médios anuais e matriz de covariância anualizada
    retornos_medios = retornos_diarios.mean() * DIAS_UTEIS_ANO
    matriz_cov = retornos_diarios.cov() * DIAS_UTEIS_ANO
    
    # Calcula volume financeiro médio diário
    preco_medio = precos.tail(126).mean()
    vol_qtd = volumes.tail(126).mean()
    volume_financeiro = (vol_qtd * preco_medio).fillna(0)

    retornos_medios, matriz_cov, volume_financeiro = limpar_dados(retornos_medios, matriz_cov, volume_financeiro)

    ativos_validos = list(retornos_medios.index)
    ret_validos = retornos_diarios[ativos_validos] 
    
    # Calcula vetor CVaR 95% e P/VP
    vetor_cvar = calcular_cvar_95(ret_validos)
    vetor_pvp = obter_pvp_ativos_otimizado(ativos_validos)

    ultimos_precos = precos[ativos_validos].ffill().iloc[-1].fillna(0.0)
    
    # Baixa benchmarks
    inicio_real = ret_validos.index[0].strftime('%Y-%m-%d')
    df_benchmarks = baixar_benchmarks(inicio_real, data_fim.strftime('%Y-%m-%d'))

    # Ajusta timezone se necessário
    if hasattr(df_benchmarks.index, 'tz') and df_benchmarks.index.tz is not None:
        df_benchmarks.index = df_benchmarks.index.tz_localize(None)
    
    df_benchmarks = df_benchmarks.reindex(ret_validos.index).ffill().bfill()

    if not df_benchmarks.empty:
        df_benchmarks = df_benchmarks / df_benchmarks.iloc[0]

    # Garante que as colunas existam
    if 'CDI' not in df_benchmarks.columns: df_benchmarks['CDI'] = 1.0
    if 'Ibovespa' not in df_benchmarks.columns: df_benchmarks['Ibovespa'] = 1.0
    if 'S&P500 (BRL)' not in df_benchmarks.columns: df_benchmarks['S&P500 (BRL)'] = df_benchmarks['Ibovespa']

    print(f"--- Inputs Prontos para {data_inicio} a {data_fim} ({len(ativos_validos)} ativos) ---")
    
    # Retorna dicionário de inputs
    return {
        'valor_total_investido': valor_total_investido,
        'retornos_medios': retornos_medios,
        'matriz_cov': matriz_cov,
        'vetor_pvp': vetor_pvp.reindex(ativos_validos).fillna(1.0),
        'vetor_cvar': vetor_cvar.reindex(ativos_validos).fillna(0.05),
        'volume_medio': volume_financeiro.reindex(ativos_validos).fillna(0),
        'ultimos_precos': ultimos_precos,
        'nomes_dos_ativos': ativos_validos,
        'n_ativos': len(ativos_validos),
        'retornos_diarios_historicos': ret_validos, 
        'df_benchmarks': df_benchmarks,
        'periodo': {'inicio': data_inicio.strftime('%Y-%m-%d'), 'fim': data_fim.strftime('%Y-%m-%d')}
    }

# Função principal para calcular inputs de otimização usando período padrão
def calcular_inputs_otimizacao(valor_total_investido):
    
    # Define período padrão
    data_fim = datetime.date.today()
    data_inicio = data_fim - datetime.timedelta(days=config.ANOS_DE_DADOS * 365.25)
    
    inputs = calcular_inputs_otimizacao_periodo(valor_total_investido, data_inicio, data_fim)
    
    # Salva debug de preços apenas na função padrão
    if inputs:
        try:
            df_debug = pd.DataFrame({
                'Ativo': inputs['ultimos_precos'].index,
                'Preco_Atual_R$': inputs['ultimos_precos'].values
            })
            df_debug.to_csv("Trabalho_OTM/valores_cotas.csv", index=False, sep=';', decimal=',')
            print(f"📄 [DEBUG] Preços atuais salvos em 'valores_cotas.csv'")
        except Exception as e:
            print(f"⚠️ Erro ao salvar CSV de debug: {e}")
    
    return inputs