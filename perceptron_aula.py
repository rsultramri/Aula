# -*- coding: utf-8 -*-

#=============================================================================
# O PERCEPTRON - AULA PRÁTICA  (versão em arquivo .py)
#=============================================================================
# Disciplina : Redes Neurais Artificiais
# Professor  : Robson de Souza Ultramari
#
# Para executar:   python perceptron_aula.py
# Requisitos:      pip install numpy matplotlib
#=============================================================================




# ===========================================================================
# # O Perceptron - Aula Prática
#
# **Disciplina:** Redes Neurais Artificiais
# **Professor:** Robson de Souza Ultramari
#
# Neste notebook implementamos do zero o perceptron de Rosenblatt (1958),
# treinamos nas portas lógicas AND e OR, desenhamos a fronteira de decisão e
# comprovamos na prática a limitação demonstrada por Minsky e Papert (1969)
# com a porta XOR.
#
# ---
#
# ### Como usar este notebook
#
# | Ação | Atalho |
# |---|---|
# | Executar a célula e ir para a próxima | `Shift + Enter` |
# | Executar e ficar na mesma célula | `Ctrl + Enter` |
# | Inserir célula abaixo | `Ctrl + M`, depois `B` |
# | Reiniciar tudo | menu *Ambiente de execução -> Reiniciar sessão* |
#
# **Importante:** execute as células **na ordem, de cima para baixo**. Cada uma
# depende do que foi definido nas anteriores. Nada precisa ser instalado:
# NumPy e Matplotlib já vêm prontos no Colab.
# ===========================================================================


# ===========================================================================
# ## Célula 1 - Bibliotecas e conjunto de dados
#
# Aqui carregamos as ferramentas e montamos a tabela-verdade da porta AND,
# que será o nosso conjunto de treinamento.
# ===========================================================================


# ---------------------------------------------------------------------
# BIBLIOTECAS
# ---------------------------------------------------------------------

# NumPy: faz contas com vetores e matrizes de uma vez só, sem precisar
# de laços. O "as np" cria um apelido curto: escrevemos np.array em vez
# de numpy.array.
import numpy as np

# Matplotlib: biblioteca de gráficos. O submódulo pyplot tem os comandos
# de desenho (plot, scatter, title...). O apelido padrão é plt.
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# CONJUNTO DE TREINAMENTO: A PORTA AND
# ---------------------------------------------------------------------

# X guarda as ENTRADAS. Cada linha é uma amostra, cada coluna é um
# atributo. Aqui temos 4 amostras (as 4 combinações possíveis) e
# 2 atributos (x1 e x2).
#
# dtype=float força os números a serem decimais. Isso importa: se
# fossem inteiros, o ajuste dos pesos (que gera 0,5) seria truncado.
X_and = np.array([[0, 0],      # amostra 1:  x1=0, x2=0
                  [0, 1],      # amostra 2:  x1=0, x2=1
                  [1, 0],      # amostra 3:  x1=1, x2=0
                  [1, 1]],     # amostra 4:  x1=1, x2=1
                 dtype=float)

# d guarda a SAÍDA DESEJADA de cada amostra, na mesma ordem de X.
# Na porta AND o resultado só é 1 quando as duas entradas são 1.
d_and = np.array([0,           # 0 AND 0 = 0
                  0,           # 0 AND 1 = 0
                  0,           # 1 AND 0 = 0
                  1])          # 1 AND 1 = 1


# ---------------------------------------------------------------------
# CONFERINDO O QUE FOI CARREGADO
# ---------------------------------------------------------------------

print("Entradas (X):")
print(X_and)

# O \n dentro do texto significa "pule uma linha" antes de escrever.
print("\nSaidas desejadas (d):", d_and)

# .shape devolve (n_linhas, n_colunas) - útil para conferir o formato.
print("\nFormato de X:", X_and.shape, "-> 4 amostras, 2 atributos")


# ===========================================================================
# ## Célula 2 - A classe Perceptron
#
# Esta é a célula central da aula. Cada método corresponde a uma fórmula
# vista na teoria:
#
# | Teoria | Método |
# |---|---|
# | $u = \mathbf{w}^T\mathbf{x} + b$ | `prever` |
# | $f(u) = 1$ se $u \geq 0$, senão $0$ | `ativacao` |
# | $e = d - \hat{y}$ e $w_i \leftarrow w_i + \eta\,e\,x_i$ | `treinar` |
#
# **Atenção:** a classe inteira tem de ficar nesta **única célula**. O Python
# não permite dividir a definição de uma classe entre células diferentes: se
# você quebrar em duas, dá erro.
# ===========================================================================


class Perceptron:
    """
    Perceptron de camada única com função de ativação degrau.

    Uma CLASSE é um molde. A partir dela criamos objetos (instâncias),
    cada um com seus próprios pesos. Assim podemos ter um perceptron
    treinado para AND e outro para OR ao mesmo tempo, sem misturar.
    """

    # =================================================================
    # CONSTRUTOR: roda automaticamente quando criamos um Perceptron
    # =================================================================
    def __init__(self, n_entradas, eta=0.1, epocas=100):
        """
        n_entradas -> quantos atributos cada amostra tem (2, no nosso caso)
        eta        -> taxa de aprendizado (o "tamanho do passo")
        epocas     -> limite de repetições, para o programa não travar

        O 'self' é o próprio objeto. Tudo que for self.alguma_coisa fica
        guardado dentro dele e continua existindo depois que o método
        termina.

        Os valores após o '=' são PADRÃO: se a pessoa não informar eta,
        ele vale 0.1 automaticamente.
        """

        # Vetor de pesos, um para cada entrada. Começamos com zeros para
        # que o resultado seja sempre o mesmo a cada execução, o que é
        # bom para a aula. Em problemas reais usam-se valores aleatórios
        # pequenos.
        self.w = np.zeros(n_entradas)

        # Viés (bias). É o que desloca a fronteira de decisão. Sem ele,
        # a reta seria obrigada a passar pela origem.
        self.b = 0.0

        # Guardamos os parâmetros para usá-los no método treinar.
        self.eta = eta
        self.epocas = epocas

        # Lista vazia que vai receber o nº de erros de cada época.
        # Serve para desenhar o gráfico de convergência mais adiante.
        self.historico = []

    # =================================================================
    # FUNÇÃO DE ATIVAÇÃO (degrau)
    # =================================================================
    def ativacao(self, u):
        """
        Converte o valor contínuo u na decisão final: 0 ou 1.
        É esta função que transforma o perceptron num classificador,
        pois ela não admite meio-termo.
        """
        # Leitura: "devolva 1 se u for maior ou igual a zero;
        # caso contrário, devolva 0".
        return 1 if u >= 0 else 0

    # =================================================================
    # PREVISÃO: soma ponderada + ativação
    # =================================================================
    def prever(self, x):
        """
        Recebe UMA amostra x e devolve a classe prevista (0 ou 1).
        """

        # np.dot faz o produto escalar: multiplica cada peso pela
        # entrada correspondente e soma tudo de uma vez.
        #   np.dot([w1, w2], [x1, x2])  =  w1*x1 + w2*x2
        # Depois somamos o viés. Isto é exatamente u = w.x + b.
        u = np.dot(self.w, x) + self.b

        # Aplica o degrau sobre o resultado.
        return self.ativacao(u)

    # =================================================================
    # TREINAMENTO: a regra de aprendizado do perceptron
    # =================================================================
    def treinar(self, X, d, verbose=True):
        """
        Ajusta w e b percorrendo o conjunto até acertar tudo.

        X       -> matriz de entradas (uma amostra por linha)
        d       -> vetor de saídas desejadas
        verbose -> se True, imprime o andamento época a época

        Devolve True se convergiu, False se estourou o limite de épocas.
        """

        # Zera o histórico, caso o mesmo objeto seja treinado de novo.
        self.historico = []

        # ----- LAÇO EXTERNO: uma volta por ÉPOCA ---------------------
        # Época = uma passagem completa por todas as amostras.
        # range(self.epocas) gera 0, 1, 2, ... até o limite.
        for epoca in range(self.epocas):

            # Contador de erros DESTA época. Se terminar em zero,
            # significa que o perceptron acertou tudo.
            erros = 0

            # ----- LAÇO INTERNO: uma volta por AMOSTRA ---------------
            # zip junta X e d em pares, para percorrer os dois ao mesmo
            # tempo: xi recebe a entrada, di a resposta certa.
            for xi, di in zip(X, d):

                # 1) O que o perceptron responde hoje?
                y = self.prever(xi)

                # 2) Qual foi o erro?
                #    Como d e y só valem 0 ou 1, o erro só pode ser:
                #      e =  0 -> acertou
                #      e = +1 -> respondeu 0 mas devia ser 1 (faltou)
                #      e = -1 -> respondeu 1 mas devia ser 0 (sobrou)
                e = di - y

                # 3) Só mexe nos pesos se errou. Esta linha é o coração
                #    do algoritmo: se acertou, não toca em nada.
                if e != 0:

                    # Regra de ajuste:  w = w + eta * e * x
                    #
                    # Repare que o ajuste é proporcional à entrada xi.
                    # Se um atributo vale 0, o peso dele NÃO muda,
                    # afinal ele não contribuiu para o erro.
                    #
                    # O sinal de e define a direção: +1 aumenta os
                    # pesos, -1 diminui.
                    self.w = self.w + self.eta * e * xi

                    # O viés se ajusta pela mesma regra, mas sem o x
                    # (é como se a entrada dele fosse sempre 1).
                    self.b = self.b + self.eta * e

                    # Registra que houve um erro nesta época.
                    erros += 1

            # ----- FIM DO LAÇO INTERNO: época concluída --------------

            # Guarda quantos erros esta época teve.
            self.historico.append(erros)

            # Mostra o andamento, se pedido.
            if verbose:
                # f-string: o que estiver entre chaves é substituído
                # pelo valor da variável.
                #   :3d  -> inteiro ocupando 3 espaços (alinha a coluna)
                #   :.2f -> número com 2 casas decimais
                # np.round(..., 2) arredonda o vetor de pesos.
                print(f"Época {epoca+1:3d} | erros: {erros} | "
                      f"w = {np.round(self.w, 2)} | b = {self.b:.2f}")

            # ----- CRITÉRIO DE PARADA -------------------------------
            # Uma época inteira sem erro nenhum significa que aprendeu.
            # O 'return' encerra o método na hora, sem fazer as épocas
            # restantes.
            if erros == 0:
                if verbose:
                    print(f"\n>>> Convergiu na época {epoca+1}.")
                return True

        # Se o laço terminou naturalmente, é porque estourou o limite
        # de épocas sem nunca zerar os erros.
        if verbose:
            print(f"\n>>> NAO convergiu em {self.epocas} épocas.")
        return False


# ===========================================================================
# ## Célula 3 - Treinando a porta AND
#
# Agora usamos a classe. Com `eta = 0.5` o resultado reproduz exatamente a
# tabela que fizemos à mão nos slides. Vale conferir linha por linha.
# ===========================================================================


# Cria um perceptron NOVO, com 2 entradas e passo 0,5.
# Neste momento os pesos ainda são zero: ele não sabe nada.
p_and = Perceptron(n_entradas=2, eta=0.5)

# Treina. Cada linha impressa é uma época.
# Acompanhe: os pesos vão mudando até estabilizar.
p_and.treinar(X_and, d_and)


# ---------------------------------------------------------------------
# TESTANDO O MODELO JÁ TREINADO
# ---------------------------------------------------------------------
print("\nTeste do modelo treinado:")

# Percorre as amostras comparando o previsto com o esperado.
for xi, di in zip(X_and, d_and):

    # .astype(int) mostra [0 1] em vez de [0. 1.]: só deixa mais limpo.
    print(f"  {xi.astype(int)} -> previsto: {p_and.prever(xi)}"
          f" | esperado: {di}")

# Os pesos finais são a "memória" do que foi aprendido.
print(f"\nPesos finais: w = {p_and.w}, b = {p_and.b}")


# ===========================================================================
# ## Célula 4 - Desenhando a fronteira de decisão
#
# Até aqui o aprendizado era um monte de números. Agora vamos **ver** o que
# aconteceu.
#
# A fronteira é a reta onde a soma ponderada vale exatamente zero:
#
# $$w_1x_1 + w_2x_2 + b = 0 \quad\Longrightarrow\quad x_2 = -\frac{w_1}{w_2}x_1 - \frac{b}{w_2}$$
#
# De um lado da reta o perceptron responde 0; do outro, 1. Treinar é girar e
# deslocar essa reta até que ela separe as duas classes.
# ===========================================================================


def plotar_fronteira(modelo, X, d, titulo):
    """
    Desenha as amostras e a reta de decisão que o modelo aprendeu.

    modelo -> um Perceptron já treinado
    X, d   -> o conjunto de dados usado
    titulo -> texto que aparece no topo do gráfico
    """

    # Cria uma figura quadrada de 5x5 polegadas.
    plt.figure(figsize=(5, 5))

    # ----- 1) DESENHAR OS PONTOS ---------------------------------
    for xi, di in zip(X, d):

        if di == 1:
            # Classe 1: um "x" vermelho.
            # xi[0] é a coordenada horizontal, xi[1] a vertical.
            # s = tamanho do marcador.
            plt.scatter(xi[0], xi[1], marker='x', s=160,
                        c='crimson', linewidths=2.5)
        else:
            # Classe 0: um círculo azul vazado.
            # facecolors='none' deixa o miolo transparente.
            plt.scatter(xi[0], xi[1], marker='o', s=140,
                        facecolors='none', edgecolors='navy',
                        linewidths=2.5)

    # ----- 2) DESENHAR A RETA DE DECISÃO -------------------------

    # Desempacota os dois pesos em variáveis separadas, só por clareza.
    w1, w2 = modelo.w
    b = modelo.b

    if w2 != 0:
        # Caso normal: dá para isolar x2 e desenhar a reta.

        # linspace cria 100 valores igualmente espaçados entre
        # -0,5 e 1,5. São os pontos da reta no eixo horizontal.
        x1 = np.linspace(-0.5, 1.5, 100)

        # Para cada x1, calcula o x2 correspondente pela fórmula.
        x2 = -(w1 / w2) * x1 - b / w2

        # 'k--' = linha preta tracejada (k de black, -- de dashed).
        plt.plot(x1, x2, 'k--', linewidth=2)

    elif w1 != 0:
        # Caso especial: w2 = 0 provocaria divisão por zero.
        # Quando isso acontece, a fronteira é uma reta VERTICAL.
        # axvline desenha uma vertical na posição indicada.
        plt.axvline(-b / w1, color='k', linestyle='--', linewidth=2)

    # Se w1 e w2 forem ambos zero não há fronteira nenhuma:
    # o modelo não aprendeu nada e simplesmente não desenhamos reta.

    # ----- 3) AJUSTES VISUAIS ------------------------------------
    plt.xlim(-0.5, 1.5)          # limites do eixo horizontal
    plt.ylim(-0.5, 1.5)          # limites do eixo vertical
    plt.xlabel('x1')             # nome do eixo horizontal
    plt.ylabel('x2')             # nome do eixo vertical
    plt.title(titulo)            # título no topo
    plt.grid(alpha=0.3)          # grade de fundo (alpha = transparência)
    plt.show()                   # exibe o gráfico


# Chama a função para o perceptron da porta AND.
# Observe: os três círculos ficam de um lado da reta, o "x" do outro.
plotar_fronteira(p_and, X_and, d_and, 'Porta AND - fronteira aprendida')


# ===========================================================================
# ## Célula 5 - A porta OR
#
# Mesmo código, apenas trocando o vetor de saídas desejadas. Compare com a
# porta AND: quantas épocas cada uma levou? Os pesos finais são parecidos?
# ===========================================================================


# Na porta OR o resultado é 1 se PELO MENOS UMA entrada for 1.
# As entradas (X_and) são as mesmas: só muda a resposta esperada.
d_or = np.array([0,    # 0 OR 0 = 0
                 1,    # 0 OR 1 = 1
                 1,    # 1 OR 0 = 1
                 1])   # 1 OR 1 = 1

# Um perceptron NOVO. Se reaproveitássemos p_and, ele começaria com os
# pesos do AND já aprendidos e o resultado seria outro.
p_or = Perceptron(n_entradas=2, eta=0.5)
p_or.treinar(X_and, d_or)

# Desenha a fronteira. Note como a reta ficou em posição diferente:
# agora ela isola apenas o ponto (0,0).
plotar_fronteira(p_or, X_and, d_or, 'Porta OR - fronteira aprendida')


# ===========================================================================
# ## Célula 6 - A porta XOR não converge
#
# Esta é a demonstração central da aula.
#
# O XOR responde 1 quando as entradas são **diferentes**. Se você marcar os
# quatro pontos no papel, vai ver que as duas classes ficam em diagonais
# opostas, e **não existe uma única reta** capaz de separá-las.
#
# O perceptron só sabe traçar retas. Logo, ele nunca vai acertar tudo.
# ===========================================================================


# Saída desejada da porta XOR: 1 quando as entradas são diferentes.
d_xor = np.array([0,    # 0 XOR 0 = 0  (iguais)
                  1,    # 0 XOR 1 = 1  (diferentes)
                  1,    # 1 XOR 0 = 1  (diferentes)
                  0])   # 1 XOR 1 = 0  (iguais)

# Limitamos a 30 épocas. Sem esse limite o algoritmo rodaria para sempre:
# é por isso que TODO código real precisa de um teto de épocas.
p_xor = Perceptron(n_entradas=2, eta=0.5, epocas=30)

# treinar devolve True ou False. Guardamos numa variável para consultar.
convergiu = p_xor.treinar(X_and, d_xor)

print("\nConvergiu?", convergiu)
print("Observe: o número de erros oscila e nunca chega a zero.")


# ===========================================================================
# ### Comparando as duas curvas
#
# O gráfico abaixo põe lado a lado o número de erros por época. A diferença
# entre as duas curvas é o resumo visual de toda a aula.
# ===========================================================================


plt.figure(figsize=(7, 4))

# historico é a lista de erros que guardamos durante o treinamento.
# range(1, len(...)+1) gera 1, 2, 3... para o eixo horizontal
# (começando em 1, porque "época zero" não existe).
plt.plot(range(1, len(p_and.historico) + 1), p_and.historico,
         'o-', label='AND (converge)')      # 'o-' = bolinha + linha

plt.plot(range(1, len(p_xor.historico) + 1), p_xor.historico,
         's-', label='XOR (não converge)')  # 's-' = quadrado + linha

plt.xlabel('época')
plt.ylabel('nº de erros')
plt.title('Erros por época: AND cai a zero, XOR nunca cai')
plt.legend()              # mostra a caixinha com os nomes das curvas
plt.grid(alpha=0.3)
plt.show()


# A fronteira do XOR: qualquer reta que o perceptron desenhe vai
# sempre deixar pelo menos um ponto do lado errado.
plotar_fronteira(p_xor, X_and, d_xor, 'Porta XOR - nenhuma reta separa')


# ===========================================================================
# ## Exercícios
#
# Resolva criando novas células abaixo (`Ctrl + M`, depois `B`).
#
# **1.** Treine o perceptron para a porta **NAND** (saídas `[1, 1, 1, 0]`).
# Compare os pesos finais com os da porta AND. O que você observa nos sinais?
#
# **2.** Na porta AND, troque a taxa de aprendizado para `0.01` e depois para
# `1.0`. Quantas épocas foram necessárias em cada caso? O passo maior sempre
# aprende mais rápido?
#
# **3.** Substitua `np.zeros(n_entradas)` por
# `np.random.uniform(-1, 1, n_entradas)` no construtor e rode cinco vezes.
# Os pesos finais são sempre iguais? E a fronteira muda de posição?
#
# **4.** Implemente a função **sinal** (saídas `-1` e `+1`) no lugar do degrau
# e refaça o treinamento da porta AND com `d = [-1, -1, -1, 1]`.
#
# **5.** Gere dois grupos de pontos aleatórios linearmente separáveis com
# `np.random.randn` e treine o perceptron sobre eles. Desenhe a fronteira.
#
# **6.** *Discussão:* por que a função degrau impede o uso de gradiente
# descendente no treinamento? Que propriedade a sigmoide tem que o degrau não
# tem? (Dica: pense na derivada.)
# ===========================================================================


# ===========================================================================
# ## Referências
#
# ROSENBLATT, F. The perceptron: a probabilistic model for information storage
# and organization in the brain. *Psychological Review*, v. 65, n. 6,
# p. 386-408, nov. 1958. DOI: 10.1037/h0042519.
#
# McCULLOCH, W. S.; PITTS, W. A logical calculus of the ideas immanent in
# nervous activity. *Bulletin of Mathematical Biophysics*, v. 5, n. 4,
# p. 115-133, 1943.
#
# NOVIKOFF, A. B. J. On convergence proofs on perceptrons. In: *Proceedings of
# the Symposium on the Mathematical Theory of Automata*, v. XII. Brooklyn:
# Polytechnic Institute of Brooklyn, 1962. p. 615-622.
#
# BLOCK, H. D. The perceptron: a model for brain functioning. I.
# *Reviews of Modern Physics*, v. 34, n. 1, p. 123-135, 1962.
#
# MINSKY, M.; PAPERT, S. *Perceptrons: an introduction to computational
# geometry*. Cambridge: MIT Press, 1969.
#
# HAYKIN, S. *Redes neurais: princípios e prática*. 2. ed. Porto Alegre:
# Bookman, 2001.
#
# GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cambridge:
# MIT Press, 2016. Disponível em: https://www.deeplearningbook.org.
# ===========================================================================
