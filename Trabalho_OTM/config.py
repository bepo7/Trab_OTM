"""
================================================
ARQUIVO DE CONFIGURAÇÃO DO PROJETO DE OTIMIZAÇÃO
================================================
"""

# --- 1. PARÂMETROS DE INVESTIMENTO ---
ANOS_DE_DADOS = 5

# --- 2. PARÂMETROS DE OTIMIZAÇÃO ---
PESO_PVP = 0.005  # Ex: 0.5 penaliza ativos caros
PESO_CVAR = 0.1
PESO_PENALIZACAO_CAIXA = 5.0 # Penaliza fortemente deixar dinheiro parado
PESO_SHARPE = 0.1 # Peso para maximização do Sharpe Ratio

# --- 3. UNIVERSO TOTAL DE ATIVOS (NOMES HUMANIZADOS) ---
UNIVERSO_ATIVOS = {
    
    "Energia e Petróleo": [
        'PETR3.SA', 'PETR4.SA', 'PRIO3.SA', 'UGPA3.SA', 'CSAN3.SA', 'VBBR3.SA', 
        'BRAV3.SA', 'RECV3.SA', 'EXXO34.SA', 'CHVX34.SA', 'B1PP34.SA', 'OXYP34.SA', 
        'SLBG34.SA', 'HALI34.SA', 'RRRP3.SA', 'ENAT3.SA', 'RAIZ4.SA'
    ],

    "Utilidades, Elétrica e Saneamento": [
        'ELET3.SA', 'ELET6.SA', 'EGIE3.SA', 'CPFE3.SA', 'CMIG4.SA', 'EQTL3.SA', 
        'TAEE11.SA', 'ALUP11.SA', 'ISAE4.SA', 'CPLE6.SA', 'NEOE3.SA', 'ENGI11.SA', 
        'AURE3.SA', 'SBSP3.SA', 'CSMG3.SA', 'SAPR11.SA', 'ENEV3.SA', 'CMIG3.SA', 
        'CPLE3.SA', 'CLSC4.SA', 'DUKB34.SA', 'NEXT34.SA', 'TRPL4.SA','AESB3.SA',  
        'MEGA3.SA'
    ],

    "Financeiro e Seguros": [
        'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 'SANB11.SA', 'BPAC11.SA', 'B3SA3.SA', 
        'ITSA4.SA', 'CXSE3.SA', 'BBSE3.SA', 'ABCB4.SA', 'BPAN4.SA', 'BRSR6.SA', 
        'PSSA3.SA', 'IRBR3.SA', 'JPMC34.SA', 'BOAC34.SA', 'CTGP34.SA', 'WFCO34.SA', 
        'GSGI34.SA', 'MSBR34.SA', 'VISA34.SA', 'MSCD34.SA', 'AXPB34.SA', 'PYPL34.SA', 
        'ROXO34.SA', 'XPBR31.SA', 'BERK34.SA', 'BMGB4.SA', 'PAGS34.SA', 'CIEL3.SA'
        ],

    "Tecnologia e Mídia": [
        'TOTS3.SA', 'LWSA3.SA', 'INTB3.SA', 'MLAS3.SA', 'AAPL34.SA', 'MSFT34.SA', 
        'GOGL34.SA', 'M1TA34.SA', 'NVDC34.SA', 'A1MD34.SA', 'ITLC34.SA', 'QCOM34.SA', 
        'AVGO34.SA', 'TSLA34.SA', 'IBMB34.SA', 'ORCL34.SA', 'SSFO34.SA', 'ADBE34.SA', 
        'NFLX34.SA', 'S1PO34.SA', 'U1BE34.SA', 'AIRB34.SA', 'Z1OM34.SA', 'CSCO34.SA', 
        'TXSA34.SA', 'CASH3.SA', 'FIQE3.SA', 'EL1V34.SA'
    ],

    "Varejo e Consumo": [
        'MGLU3.SA', 'BHIA3.SA', 'LREN3.SA', 'AZZA3.SA', 'ALOS3.SA', 'ASAI3.SA', 
        'GMAT3.SA', 'PETZ3.SA', 'RADL3.SA', 'RAIL3.SA', 'VIVA3.SA', 'ZAMP3.SA', 
        'ABEV3.SA', 'MDIA3.SA', 'CAML3.SA', 'SMTO3.SA', 'BEEF3.SA', 'MRFG3.SA', 
        'BRFS3.SA', 'AMZO34.SA', 'MELI34.SA', 'WALM34.SA', 'TGTB34.SA', 'HOME34.SA', 
        'NIKE34.SA', 'MCDC34.SA', 'COCA34.SA', 'PEPB34.SA', 'PGCO34.SA', 'COLG34.SA', 
        'JNJB34.SA', 'ARZZ3.SA', 'SOMA3.SA', 'GUAR3.SA', 'CEAB3.SA' 
    ],

    "Saúde e Farmacêutica": [
        'HAPV3.SA', 'RDOR3.SA', 'FLRY3.SA', 'QUAL3.SA', 'ODPV3.SA', 'VVEO3.SA',
        'PFIZ34.SA', 'MRCK34.SA', 'ABBV34.SA', 'LILY34.SA', 'UNHH34.SA', 'BMYB34.SA',
        'M1RN34.SA', 'GILD34.SA', 'A1ZN34.SA', 'PNVL3.SA', 'MATD3.SA'
    ],

    "Materiais Básicos e Industrial": [
        'VALE3.SA', 'GGBR4.SA', 'GOAU4.SA', 'CSNA3.SA', 'USIM5.SA', 'SUZB3.SA', 
        'KLBN11.SA', 'BRAP4.SA', 'CMIN3.SA', 'DXCO3.SA', 'FESA4.SA', 'UNIP6.SA',
        'WEGE3.SA', 'EMBR3.SA', 'POMO4.SA', 'RAPT4.SA', 'TASA4.SA', 'RIOT34.SA', 
        'AALL34.SA', 'FCXO34.SA', 'GEOO34.SA', 'CATP34.SA', 'BOEI34.SA', 'LMTB34.SA', 
        'MMMC34.SA', 'DEEC34.SA', 'HONB34.SA', 'UPSS34.SA', 'FDXB34.SA', 'CBAV3.SA',  
        'RANI3.SA', 'SHUL4.SA'
    ],

    "Imobiliário e Construção": [
        'CYRE3.SA', 'EZTC3.SA', 'MRVE3.SA', 'DIRR3.SA', 'TEND3.SA', 'JHSF3.SA',
        'GFSA3.SA', 'HBOR3.SA', 'CURY3.SA', 'PLPL3.SA', 'IGTI11.SA', 'MULT3.SA', 
        'ALOS3.SA', 'LOGG3.SA', 'A1MT34.SA', 'P1LD34.SA', 'C1CI34.SA', 'R1IN34.SA', 
        'E1QR34.SA', 'SIMN34.SA', 'LAVV3.SA', 'EVEN3.SA', 'MELK3.SA'   
    ],

    "Mercado Europeu e Asiático": [
        'XINA11.SA', 'ASIA11.SA', 'SAPP34.SA', 'ASML34.SA', 'H1SB34.SA', 'BABA34.SA', 
        'TSMC34.SA', 'SNEC34.SA', 'TMCO34.SA', 'BIDU34.SA', 'JDCO34.SA'
    ],

    "Criptoativos": [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 
        'AVAX-USD', 'LINK-USD', 'DOT-USD', 'MATIC-USD', 'LTC-USD', 'BCH-USD', 
        'UNI7083-USD', 'AAVE-USD', 'ATOM-USD', 'XLM-USD', 'TRX-USD', 'ETC-USD',
        'SHIB-USD', 'ALGO-USD', 'NEAR-USD'
    ],

    "ETFS e Índices Globais": [
        'IVVB11.SA', 'SPXI11.SA', 'NASD11.SA', 'BOVA11.SA', 'SMAL11.SA', 'BRAX11.SA',
        'ECOO11.SA', 'GOVE11.SA', 'MATB11.SA', 'FIND11.SA', 'DIVO11.SA', 'PIBB11.SA',
        'GOLD11.SA', 'HASH11.SA', 'QBTC11.SA', 'QETH11.SA', 'TECK11.SA', 'DNAI11.SA',
        'EURP11.SA', 'XINA11.SA'
    ],

    "FIIs (Fundos Imobiliários)": [
        'KNCR11.SA', 'KNIP11.SA', 'HGCR11.SA', 'MXRF11.SA', 'CPTS11.SA', 'RECR11.SA', 
        'IRDM11.SA', 'VRTA11.SA', 'CVBI11.SA', 'MCCI11.SA', 'VGIP11.SA', 'RBRR11.SA',
        'TGAR11.SA', 'DEVA11.SA', 'HCTR11.SA', 'VSLH11.SA', 'KNSC11.SA', 'RBRY11.SA',
        'HGLG11.SA', 'BTLG11.SA', 'XPLG11.SA', 'VILG11.SA', 'BRCO11.SA', 'LVBI11.SA', 
        'GGRC11.SA', 'SDIL11.SA', 'ALZR11.SA', 'RBRL11.SA', 'PATL11.SA', 'XPML11.SA', 
        'VISC11.SA', 'HGBS11.SA', 'HSML11.SA', 'TRXF11.SA', 'RBRP11.SA', 'KNRI11.SA', 
        'BRCR11.SA', 'HGRE11.SA', 'JSRE11.SA', 'PVBI11.SA', 'RCRB11.SA', 'VINO11.SA', 
        'RECT11.SA', 'TEPP11.SA', 'HGRU11.SA', 'KNHY11.SA', 'HFOF11.SA', 'RBRF11.SA', 
        'XPSF11.SA', 'KISU11.SA', 'VGHF11.SA', 'URPR11.SA', 'HABT11.SA', 'RZTR11.SA', 
        'GAME11.SA', 'VGAF11.SA'
    ],

    "Renda Fixa (ETFs)": [
        'B5P211.SA', 'IMAB11.SA', 'IRFM11.SA', 'LFTS11.SA', 'LTNB11.SA', 'IB5M11.SA'
    ]
}

# --- 4. LISTA DE EXCLUSÃO ---
TICKERS_COM_FALHA_YF = [
    'GOLL4.SA', 'AMBP3.SA', 'OIBR3.SA', 'RCSL3.SA', 'SLED4.SA', 'PDGR3.SA',
    'JBSS3.SA', 'CRFB3.SA', 'NTCO3.SA', 'MALL11.SA', 'BCFF11.SA', 'EURP11.SA',
    'BHPG34.SA', 'S1OE34.SA', 'SHEL34.SA', 'ULVR34.SA', 'SBUB34.SA',
    'VGAF11.SA', 'AESB3.SA', 'MEGA3.SA', 'ARZZ3.SA', 'CIEL3.SA', 
    'TRPL4.SA', 'SOMA3.SA', 'EL1V34.SA', 'ENAT3.SA', 'RRRP3.SA', 'BRFS3.SA',
    'MRFG3.SA', 'CVBI11.SA', 'USIM5.SA'
]

def obter_universo_completo():
    lista_completa = []
    for categoria in UNIVERSO_ATIVOS:
        lista_completa.extend(UNIVERSO_ATIVOS[categoria])
    lista_unica = sorted(list(set(lista_completa)))
    lista_limpa = [t for t in lista_unica if t not in TICKERS_COM_FALHA_YF]
    print(f"--- Config: Universo expandido de {len(lista_limpa)} ativos carregado ---")
    return lista_limpa

# --- NOVA FUNÇÃO AUXILIAR PARA MAPEAMENTO ---
def obter_mapa_setores_ativos():
    """
    Retorna um dicionário {Nome_Setor: [Lista_Ativos_Limpa]}
    """
    mapa = {}
    for setor, ativos in UNIVERSO_ATIVOS.items():
        # Filtra os ativos que estão na lista de falha
        ativos_limpos = [a for a in ativos if a not in TICKERS_COM_FALHA_YF]
        if ativos_limpos:
             mapa[setor] = ativos_limpos
    return mapa

UNIVERSO_COMPLETO = obter_universo_completo()