import numpy as np
from pymoo.core.problem import Problem
import config 

class OtimizacaoPortfolio(Problem):
    def __init__(self, retornos_medios, matriz_cov, 
                 vetor_pvp, vetor_cvar, 
                 volume_medio, valor_investido,
                 risco_maximo_usuario, lambda_aversao_risco,
                 nomes_ativos=None, mapa_setores=None, setores_proibidos=None,
                 teto_maximo_ativo=0.30, teto_maximo_setor=1.0, verbose=True):
        
        self.retornos_medios = retornos_medios
        self.matriz_cov = matriz_cov
        self.vetor_pvp = vetor_pvp.values     
        self.vetor_cvar = vetor_cvar.values   
        
        self.risco_maximo_usuario = risco_maximo_usuario
        self.lambda_aversao_risco = lambda_aversao_risco
        
        self.teto_maximo_ativo = teto_maximo_ativo
        self.teto_maximo_setor = teto_maximo_setor
        n_ativos = len(retornos_medios)
        
        xl = np.full(n_ativos, 0.0)
        
        # --- CÁLCULO DOS LIMITES SUPERIORES (LIQUIDEZ + TETO) ---
        
        # 1. Blindagem: Garante que não tem NaNs no volume antes de calcular
        vol_values = np.nan_to_num(volume_medio.values, nan=0.0)
        teto_financeiro_liquidez = 0.1 * vol_values
        
        # 2. Converte para teto em peso
        inv = valor_investido if valor_investido > 0 else 1.0
        max_peso_liquidez = teto_financeiro_liquidez / inv
        
        # 3. Lógica de Limite em Etapas (Para evitar erro do numpy)
        # O limite individual (xu) deve ser o MENOR entre:
        # A. Teto do Ativo (ex: 30%)
        # B. Liquidez do Ativo
        # C. Teto do Setor (ex: 10%) -> Essencial! Um ativo não pode ser maior que seu setor.
        
        xu = np.minimum(teto_maximo_ativo, max_peso_liquidez)
        xu = np.minimum(xu, teto_maximo_setor) # <--- APLICAÇÃO DO TETO SETORIAL
        
        # --- SETORES PROIBIDOS ---
        if setores_proibidos and nomes_ativos and mapa_setores:
            ticker_to_idx = {ticker: i for i, ticker in enumerate(nomes_ativos)}
            for setor in setores_proibidos:
                if setor in mapa_setores:
                    for ativo in mapa_setores[setor]:
                        # Remove espaços em branco extras para garantir o match
                        ativo_limpo = ativo.strip()
                        # Procura o ativo na lista de nomes (também limpa)
                        # Otimização: Tenta acesso direto primeiro, se falhar, busca linear
                        if ativo_limpo in ticker_to_idx:
                            idx = ticker_to_idx[ativo_limpo]
                            xu[idx] = 0.0
                            
        # Garante que não ficou nada negativo
        xu = np.maximum(0.0, xu)

        # CONFIGURAÇÃO DO PYMOO: 
        # n_constr=2 (1 Risco + 1 Orçamento Máximo)
        # n_eq_constr=0 (Nenhuma igualdade, para permitir Caixa)
        super().__init__(n_var=n_ativos, n_obj=1, n_constr=2, n_eq_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # 1. Retorno Esperado
        retorno_port = x.dot(self.retornos_medios)
        
        # 2. Variância (Risco)
        variancia = np.einsum('...i,ij,...j->...', x, self.matriz_cov, x)
        risco_vol = np.sqrt(np.maximum(variancia, 1e-12))
        
        # 3. P/VP e CVaR (Objetivos Secundários)
        pvp_port = x.dot(self.vetor_pvp)
        cvar_port = x.dot(self.vetor_cvar)
        
        # 4. Penalidade por Caixa Não Investido
        soma_pesos = np.sum(x, axis=1)
        caixa_nao_investido = np.maximum(0.0, 1.0 - soma_pesos)
        penalidade_caixa = config.PESO_PENALIZACAO_CAIXA * caixa_nao_investido

        # Função Objetivo (Multiobjetivo Scalarizado)
        # Minimizamos: (Risco * Lambda) - Retorno + Penalidades
        obj = (self.lambda_aversao_risco * variancia) - retorno_port \
              + (config.PESO_PVP * pvp_port) \
              + (config.PESO_CVAR * cvar_port) \
              + penalidade_caixa
        
        out["F"] = obj
        
        # --- RESTRIÇÕES ---
        
        # G1: Risco <= Risco Máximo do Usuário
        g1 = risco_vol - self.risco_maximo_usuario
        
        # G2: Orçamento <= 100% (Desigualdade)
        # Permite que a soma seja menor que 1.0 (gerando "Caixa"), mas nunca maior.
        # Expressão: soma - 1.0 <= 0
        g2 = np.sum(x, axis=1) - 1.0
        
        # Empilha as restrições na saída "G"
        out["G"] = np.column_stack([g1, g2])