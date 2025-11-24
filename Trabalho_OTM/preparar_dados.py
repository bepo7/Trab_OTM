import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from bcb import sgs
import warnings

# --- 1. IMPORTAR CONFIGURAÇÕES DO USUÁRIO ---
import config

# --- 2. PARÂMETROS DE IMPLEMENTAÇÃO ---
BENCHMARK_MERCADO = '^BVSP'
CDI_CODIGO_BCB = 4389 # CDI Anualizado base 252
TAXA_LIVRE_RISCO_FALLBACK = 0.15 
DIAS_UTEIS_ANO = 252

# Suprimir warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# --- 3. FUNÇÕES DE COLETA DE DADOS ---

def baixar_taxa_livre_de_risco(data_inicio_str, data_fim_str):
    """ 
    Tenta baixar a série histórica do CDI (Cód: 4389) do BCB 
    apenas para referência de métricas (Sharpe, etc).
    """
    print(f"Tentando baixar Taxa CDI (Cód: {CDI_CODIGO_BCB}) do BCB...")
    try:
        cdi_df = sgs.get({'cdi': CDI_CODIGO_BCB}, 
                           start=data_inicio_str, 
                           end=data_fim_str)
        
        cdi_df = cdi_df.dropna()
        if cdi_df.empty:
            raise ValueError("Dados do CDI vieram vazios.")

        ultimo_valor = cdi_df['cdi'].iloc[-1]
        taxa_real = ultimo_valor / 100.0
        
        print(f"Sucesso: CDI real referência obtido ({taxa_real:.2%})")
        return taxa_real
        
    except Exception as e:
        print(f"Falha ao baixar CDI do BCB: {e}. Usando fallback de {TAXA_LIVRE_RISCO_FALLBACK:.2%}")
        return TAXA_LIVRE_RISCO_FALLBACK

def baixar_dados_ativos(lista_de_tickers, data_inicio, data_fim):
    """
    Baixa os preços de fechamento ajustados para a lista de tickers.
    Agora inclui LFTS11.SA naturalmente.
    """
    
    if not lista_de_tickers:
        print("Nenhum ativo para baixar do yfinance.")
        return pd.DataFrame() 
    
    print(f"Baixando dados históricos de {len(lista_de_tickers)} ativos (reais)...")
    try:
        # Tenta baixar
        dados = yf.download(lista_de_tickers, 
                            start=data_inicio, 
                            end=data_fim, 
                            progress=True,
                            auto_adjust=True)
        
        if dados.empty:
            print("Erro: Nenhum dado foi baixado.")
            return None
            
        # Extrai Fechamento (Compatibilidade com versões novas e antigas do YF)
        if 'Close' in dados.columns:
            precos = dados['Close']
        else:
            precos = dados
        
        # Se baixou apenas 1 ativo, garante que é DataFrame
        if len(lista_de_tickers) == 1 and isinstance(precos, pd.Series):
            precos = precos.to_frame(lista_de_tickers[0])
            
        # Limpeza Inicial de Colunas Vazias
        precos = precos.dropna(axis=1, how='all')
        
        # Preenchimento de Falhas (Finais de semana/Feriados locais)
        precos = precos.ffill().bfill()
        
        # Limpeza Final
        precos = precos.dropna(axis=1, how='all')
        
        if precos.empty:
            print("Erro: Dados vazios após limpeza.")
            return None
            
        print(f"Sucesso: Dados de {len(precos.columns)} ativos válidos foram baixados e alinhados.")
        return precos

    except Exception as e:
        print(f"Erro crítico durante o download do yfinance: {e}")
        return None

# --- NOVA FUNÇÃO DE LIMPEZA ROBUSTA ---
def limpar_matriz_covariancia_e_retornos(retornos_medios, matriz_cov):
    """
    Remove ativos que causam instabilidade matemática (NaN/Inf).
    """
    ativos_removidos = []
    
    # 1. Remover Ativos com Média Inválida
    mask_mu_bad = ~np.isfinite(retornos_medios)
    if mask_mu_bad.any():
        bad_assets = retornos_medios[mask_mu_bad].index.tolist()
        ativos_removidos.extend(bad_assets)
        retornos_medios = retornos_medios.drop(bad_assets)
        matriz_cov = matriz_cov.drop(index=bad_assets, columns=bad_assets)

    # 2. Remover Ativos com Variância Inválida
    diag = np.diag(matriz_cov)
    mask_var_bad = ~np.isfinite(diag)
    if mask_var_bad.any():
        indices_ruins = np.where(mask_var_bad)[0]
        nomes_ruins = matriz_cov.index[indices_ruins].tolist()
        ativos_removidos.extend(nomes_ruins)
        retornos_medios = retornos_medios.drop(nomes_ruins)
        matriz_cov = matriz_cov.drop(index=nomes_ruins, columns=nomes_ruins)

    # 3. Limpeza Iterativa de Covariância
    while matriz_cov.isnull().values.any():
        contagem_nans = matriz_cov.isnull().sum(axis=1)
        pior_ativo = contagem_nans.idxmax()
        
        ativos_removidos.append(pior_ativo)
        retornos_medios = retornos_medios.drop(pior_ativo)
        matriz_cov = matriz_cov.drop(index=pior_ativo, columns=pior_ativo)
    
    if ativos_removidos:
        print(f"LIMPEZA ROBUSTA: Removidos {len(ativos_removidos)} ativos problemáticos (NaN/Inf):")
        print(f" -> {ativos_removidos}")
        
    return retornos_medios, matriz_cov

# --- 4. FUNÇÃO PRINCIPAL DE PREPARAÇÃO ---

def calcular_inputs_otimizacao(valor_total_investido):
    
    lista_de_ativos_config = config.UNIVERSO_COMPLETO
    if not lista_de_ativos_config:
        print("Erro: 'UNIVERSO_COMPLETO' no config.py está vazio.")
        return None

    # --- 1. Definir Datas ---
    data_fim = datetime.date.today()
    data_inicio = data_fim - datetime.timedelta(days=config.ANOS_DE_DADOS * 365.25)
    
    data_fim_str = data_fim.strftime('%Y-%m-%d')
    data_inicio_str = data_inicio.strftime('%Y-%m-%d')
    
    # --- 2. Baixar Taxa de Risco (Referência) ---
    taxa_livre_de_risco = baixar_taxa_livre_de_risco(data_inicio_str, data_fim_str)
    
    # --- 3. Baixar Dados dos Ativos (Incluindo LFTS11) ---
    precos = baixar_dados_ativos(lista_de_ativos_config, data_inicio, data_fim)
    
    if precos is None or precos.empty:
        print("Erro: Nenhum dado de preço foi obtido.")
        return None

    # --- 4. Calcular Retornos e Matriz de Covariância ---
    retornos_diarios = precos.pct_change()
    
    if np.isinf(retornos_diarios.values).any():
        print("Aviso: Valores infinitos detectados nos retornos. Limpando...")
        retornos_diarios = retornos_diarios.replace([np.inf, -np.inf], np.nan)
        
    retornos_diarios = retornos_diarios.dropna()

    if retornos_diarios.empty:
        print("Erro: Dados insuficientes após cálculo de retornos.")
        return None
    
    print("Calculando μ (Retornos Médios) e Σ (Matriz de Covariância)...")
    retornos_medios_anuais = retornos_diarios.mean() * DIAS_UTEIS_ANO
    matriz_cov_anual = retornos_diarios.cov() * DIAS_UTEIS_ANO

    # Limpeza Robusta
    retornos_medios_anuais, matriz_cov_anual = limpar_matriz_covariancia_e_retornos(
        retornos_medios_anuais, 
        matriz_cov_anual
    )
    
    nomes_dos_ativos_validos = list(retornos_medios_anuais.index)
        
    print("\n--- Inputs de Otimização Prontos ---")
    
    return {
        'valor_total_investido': valor_total_investido, 
        'retornos_medios': retornos_medios_anuais,
        'matriz_cov': matriz_cov_anual,
        'taxa_livre_de_risco': taxa_livre_de_risco, 
        'nomes_dos_ativos': nomes_dos_ativos_validos,
        'n_ativos': len(nomes_dos_ativos_validos)
    }

if __name__ == '__main__':
    print("--- TESTE RÁPIDO ---")
    inputs = calcular_inputs_otimizacao(1000)
    if inputs:
        print(f"Sucesso! {inputs['n_ativos']} ativos prontos para o GA.")