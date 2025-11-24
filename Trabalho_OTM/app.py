from flask import Flask, render_template, request, jsonify, url_for
import os
import time
import traceback
import numpy as np

# --- CONFIGURAÇÃO DE CAMINHOS ABSOLUTOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'plots')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'site')

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATE_DIR)

# --- IMPORTS LOCAIS ---
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

# --- FUNÇÃO AUXILIAR PARA FORMATAR DADOS ---
def formatar_dados_para_frontend(ticker_list, peso_array, valor_investido):
    """Converte arrays de pesos em lista de objetos para o JSON"""
    alocacao = []
    for i, ticker in enumerate(ticker_list):
        peso = peso_array[i]
        if peso > 1e-4: # Filtro de 0.01%
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

        # 2. Execução: ALGORITMO GENÉTICO (Base)
        print(">> Rodando GA...")
        res_ga = otimizar.rodar_otimização(inputs, risco_teto, lambda_risco, setores_proibidos)
        
        if res_ga is None:
            return jsonify({'sucesso': False, 'erro': 'GA não convergiu.'}), 400

        # 3. Execução: GUROBI (Refinamento)
        print(">> Rodando Gurobi...")
        pesos_iniciais_ga = res_ga['pesos_finais']
        
        res_gurobi = otm_gurobi_setores.resolver_com_gurobi_setores(
            inputs, lambda_risco, risco_teto, 
            warm_start_pesos=pesos_iniciais_ga, 
            setores_proibidos=setores_proibidos
        )

        # 4. Geração de Imagens
        timestamp = int(time.time())
        nome_ga = 'grafico_ga.png'
        nome_gu = 'grafico_gurobi.png'
        
        path_ga = os.path.join(STATIC_DIR, nome_ga)
        path_gu = os.path.join(STATIC_DIR, nome_gu)

        plot.rodar_visualizacao_dupla(inputs, res_ga, res_gurobi, path_ga, path_gu)

        # 5. PREPARAR DADOS DO GA (Algoritmo Genético)
        row_ga = res_ga['dataframe_resultado'].iloc[0]
        cols_meta = ['Risco_Alvo', 'Risco_Encontrado_Anual', 'Retorno_Encontrado_Anual']
        pesos_ga_series = row_ga.drop(cols_meta, errors='ignore')
        pesos_ga_final = pesos_ga_series.reindex(nomes_ativos).fillna(0.0).values

        data_ga = {
            'metricas': {
                'valor_investido': valor_investir,
                'retorno_aa': res_ga['retorno_final'] * 100,
                'risco_aa': res_ga['risco_final'] * 100,
                'score': res_ga['funcao_objetivo']
            },
            'alocacao': formatar_dados_para_frontend(nomes_ativos, pesos_ga_final, valor_investir),
            'grafico_url': url_for('static', filename=nome_ga) + f'?t={timestamp}'
        }

        # 6. PREPARAR DADOS DO GUROBI (Se existir)
        data_gurobi = None
        if res_gurobi:
            data_gurobi = {
                'metricas': {
                    'valor_investido': valor_investir,
                    'retorno_aa': res_gurobi['retorno'] * 100,
                    'risco_aa': res_gurobi['risco'] * 100,
                    'score': res_gurobi['obj']
                },
                'alocacao': formatar_dados_para_frontend(nomes_ativos, res_gurobi['pesos'], valor_investir),
                'grafico_url': url_for('static', filename=nome_gu) + f'?t={timestamp}'
            }

        # 7. Retornar Ambos
        return jsonify({
            'sucesso': True,
            'ga': data_ga,
            'gurobi': data_gurobi
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'sucesso': False, 'erro': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)