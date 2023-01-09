'''
Metodos para otimização em problemas não lineares irrestritos
'''

import numpy as np
np.set_printoptions(suppress=True, precision=8, threshold=np.inf, linewidth=np.inf)
from sympy import *

class OtimizacaoNLP():
    '''
    Implementa funções auxiliares e modelos para minimização de uma função
    não linear a partir de um ponto inicial.
    
    Parametros:
    func = função a ser minimizada em formato de string
    vars = lista com todas as variáveis da função de entrada
    init_p = lista com os valores das variáveis no ponto inicial do algoritmo
    '''   
    def __init__(self, func, vars, init_p):  
        self.func = parse_expr(func)
        self.vars = symbols(vars)
        self.nvars = len(vars)  # Numero de variáveis da função func
        self.init_p = init_p
        
    def __getitem__(self, index):
        return self.gradiente[index]
    
    def gradiente(self): 
        ''' Matriz Gradiente de func. Retorna a matriz de simbolos como numpy array. '''
        grad_func = np.zeros((self.nvars, 1), dtype=object)
        
        for lin in range(grad_func.shape[0]):
            grad_func[lin] = diff(self.func, self.vars[lin])
  
        return grad_func
    
    def hessiana(self):
        ''' Matriz Hessiana de func. Retorna a matriz de simbolos como numpy array. '''
        hess_func = np.zeros((self.nvars, self.nvars), dtype=object)
        
        for lin in range(hess_func.shape[0]):  # Para cada linha do gradiente faz a derivada parcial sobre cada variável
            for col in range(hess_func.shape[1]):
                hess_func[lin][col] = diff(self.gradiente()[lin][0], self.vars[col])
            
        return hess_func
    
    def calcular_func(self, func = None, vars = None, p = None): 
        ''' Calcula valor numérico de uma função qualquer dado um ponto p.
            Se nenhuma função é provida, calcula func num ponto p.
        '''          
        if func is None:
            func = self.func
        if vars is None:
            vars = self.vars
        if p is None:
            p = self.init_p
        if isinstance(p, np.matrix):
            if p.shape[1] == 1:
                p = np.matrix.tolist(p.T)[0]
            elif p.shape[0] == 1:
                p = np.matrix.tolist(p)[0]
            else:
                raise ValueError
        if isinstance(func, str):
            func = parse_expr(func)
            
        result = func.subs(zip(vars, p)).evalf()
        return result
    
    def gradiente_p(self, p): 
        ''' Valor da matriz Gradiente de func no ponto p. Retorna a matriz como numpy array. '''
        grad_p = np.zeros((self.gradiente().shape[0], 1))

        for lin in range(grad_p.shape[0]):
            grad_p[lin] = self.calcular_func(self.gradiente()[lin][0], self.vars, p)
        return grad_p
    
    def hessiana_p(self, p): 
        ''' Valor da matriz Hessiana de func no ponto p. Retorna a matriz como numpy array. '''
        hess_p = np.zeros(self.hessiana().shape)
        for lin in range(hess_p.shape[0]):
            for col in range(hess_p.shape[1]):
                hess_p[lin][col] = self.calcular_func(self.hessiana()[lin][col], self.vars, p)       
        return hess_p

    def passo_armijo(self, p, direcao, eta_param, gamma_param):
        ''' Realiza busca de armijo para encontrar o salto usado nos métodos de otimização '''
        if isinstance(p, list):
            p = np.matrix(p)
        if isinstance(direcao, list):
            direcao = np.matrix(direcao)

        t = 1  # Salto Inicial
        const1 = self.calcular_func(self.func, self.vars, p)
        const2 = eta_param * np.linalg.det(np.matrix(self.gradiente_p(p)).T * direcao)
        
        cont = 1  # Contagem de chamadas de Armijo 
        while self.calcular_func(self.func, self.vars, p + t * direcao) > const1 + t * const2:
            t = gamma_param * t
            cont += 1

        return (t, cont)
        
    def metodo_newton(self, verbose = False):
        ''' Aplicação do Método de Newton para otimização de funções '''
        p_atual = np.matrix(self.init_p).T
        cont_passos = 0
        cont_armijo = 0
        decrem = 100000
        dir_sub = 0  # Variável que vai manter registro de direções falhas (de subida)
        
        if verbose:
            print(f"\nComeçando Metodo de Newton com Ponto Inicial {p_atual}, onde a F.O. "
                  f"vale {self.calcular_func(self.func, self.vars, p_atual):.8f}\n")
        
        while (decrem/2 > 1e-06 and cont_passos < 500):
            # Definindo direção d[k]
            m_hessiana = np.matrix(self.hessiana_p(p_atual))
            m_gradiente = np.matrix(self.gradiente_p(p_atual))
            try:
                direcao = - np.linalg.inv(m_hessiana)*m_gradiente  # Passo de Newton
            except np.linalg.LinAlgError:
                if verbose:
                    print(f"Erro: A matriz Hessiana não é inversível.\n")
                raise
            else:
                
                # Analise da direção d[k], que deve ser menor que zero
                direcao_analise = np.linalg.det(m_gradiente.T*direcao)
                if direcao_analise > 0:
                    dir_sub += 1
                    if verbose:
                        print(f"Warning: A direção de deslocamento é de subida.\n")
                
                # Computando Decremento de Newton (ao quadrado)
                decrem = np.linalg.det ( m_gradiente.T * np.linalg.inv(m_hessiana) * m_gradiente )

                # Obtendo o passo t[k] e o x[k+1]
                passo_t = self.passo_armijo(p_atual, direcao, 0.25, 0.8)[0]  # Passo pela Busca de Armijo
                cont_armijo += self.passo_armijo(p_atual, direcao, 0.25, 0.8)[1]

                p_atual = p_atual + passo_t * direcao
                cont_passos += 1

                if verbose:
                    print(f"Passo Nº {cont_passos}, Ponto Atual = {p_atual}, "
                        f"Valor da F.O.: {self.calcular_func(self.func, self.vars, p_atual):.8f}\n")

        if verbose:
            print(f"Finalizado em {cont_passos} passos, com Ponto Ótimo = {p_atual} e "
                  f"Valor Ótimo da F.O.: {self.calcular_func(self.func, self.vars, p_atual):.8f} "
                  f"\nObtidos {dir_sub} pontos onde a direção era de subida.\n")

        return (p_atual, self.calcular_func(self.func, self.vars, p_atual), cont_passos, cont_armijo)
    
    def metodo_gradiente(self, verbose = False):
        ''' Aplicação do Método do Gradiente para otimização de funções '''
        p_atual = np.matrix(self.init_p).T
        cont_passos = 0
        cont_armijo = 0

        m_gradiente = np.matrix(self.gradiente_p(p_atual)) # Gradiente no ponto inicial

        if verbose:
            print(f"\nComeçando Metodo do Gradiente com Ponto Inicial {p_atual}, onde a F.O. "
                  f"vale {self.calcular_func(self.func, self.vars, p_atual):.8f}\n")

        while (np.all(np.abs(m_gradiente) > 1e-06) and cont_passos < 500):  # Repetir enquanto o gradiente não for aproximadamente zero
            # Definindo direção d[k]
            m_gradiente = np.matrix(self.gradiente_p(p_atual))
            direcao = - m_gradiente

            # Obtendo o passo t[k] e o x[k+1]
            passo_t = self.passo_armijo(p_atual, direcao, 0.25, 0.8)[0]  # Passo pela Busca de Armijo
            cont_armijo += self.passo_armijo(p_atual, direcao, 0.25, 0.8)[1]  

            p_atual = p_atual + passo_t * direcao
            cont_passos += 1

            if verbose:
                print(f"Passo Nº {cont_passos}, Ponto Atual = {p_atual}, "
                      f"Valor da F.O.: {self.calcular_func(self.func, self.vars, p_atual):.8f}\n")

        if verbose:
            print(f"Finalizado em {cont_passos} passos, com Ponto Ótimo = {p_atual} e "
                  f"Valor Ótimo da F.O.: {self.calcular_func(self.func, self.vars, p_atual):.8f}\n")

        return (p_atual, self.calcular_func(self.func, self.vars, p_atual), cont_passos, cont_armijo)

    def metodo_dfp(self, verbose=False):
        ''' Aplicação do Método DFP (um Método de Quase-Newton) para otimização de funções '''
        p_atual = np.matrix(self.init_p).T
        cont_passos = 0
        cont_armijo = 0

        m_gradiente = np.matrix(self.gradiente_p(p_atual)) # Gradiente no ponto inicial
        h = np.identity(self.nvars) # H inicial é a matriz identidade

        if verbose:
            print(f"\nComeçando Metodo DFP com Ponto Inicial {p_atual}, onde a F.O. "
                  f"vale {self.calcular_func(self.func, self.vars, p_atual):.8f}\n")

        while (np.all(np.abs(m_gradiente) > 1e-06) and cont_passos < 500):
            # Definindo direção d[k]
            m_gradiente = np.matrix(self.gradiente_p(p_atual))
            direcao = - h * m_gradiente
            
            # Obtendo o passo t[k] e o x[k+1]
            passo_t = self.passo_armijo(p_atual, direcao, 0.25, 0.8)[0]  # Passo pela Busca de Armijo
            cont_armijo += self.passo_armijo(p_atual, direcao, 0.25, 0.8)[1]  
            
            p_prox = p_atual + passo_t * direcao
            
            # Determinando H[k+1]
            p = p_prox - p_atual
            q = np.matrix(self.gradiente_p(p_prox)) - m_gradiente          
            h = (h + np.linalg.det(p * p.T)/np.linalg.det(p.T * q) 
                 - np.linalg.det(h * q * q.T * h)/np.linalg.det(q.T * h * q))

            p_atual = p_prox
            cont_passos += 1

            if verbose:
                print(f"Passo Nº {cont_passos}, Ponto Atual = {p_atual}, "
                      f"Valor da F.O.: {self.calcular_func(self.func, self.vars, p_atual):.8f}\n")

        if verbose:
            print(f"Finalizado em {cont_passos} passos, com Ponto Ótimo = {p_atual} e "
                  f"Valor Ótimo da F.O.: {self.calcular_func(self.func, self.vars, p_atual):.8f}\n")

        return (p_atual, self.calcular_func(self.func, self.vars, p_atual), cont_passos, cont_armijo)
    
    def metodo_bfgs(self, verbose=False):
        ''' Aplicação do Método BFGS (um Método de Quase-Newton) para otimização de funções '''
        p_atual = np.matrix(self.init_p).T
        cont_passos = 0
        cont_armijo = 0

        m_gradiente = np.matrix(self.gradiente_p(p_atual)) # Gradiente no ponto inicial
        h = np.identity(self.nvars) # H inicial é a matriz identidade

        if verbose:
            print(f"\nComeçando Metodo BFGS com Ponto Inicial {p_atual}, onde a F.O. "
                  f"vale {self.calcular_func(self.func, self.vars, p_atual):.8f}\n")

        while (np.all(np.abs(m_gradiente) > 1e-06) and cont_passos < 500):
            # Definindo direção d[k]
            m_gradiente = np.matrix(self.gradiente_p(p_atual))
            direcao = - h * m_gradiente
            
            # Obtendo o passo t[k] e o x[k+1]
            passo_t = self.passo_armijo(p_atual, direcao, 0.25, 0.8)[0]  # Passo pela Busca de Armijo
            cont_armijo += self.passo_armijo(p_atual, direcao, 0.25, 0.8)[1]  
            
            p_prox = p_atual + passo_t * direcao
            
            # Determinando H[k+1]
            p = p_prox - p_atual
            q = np.matrix(self.gradiente_p(p_prox)) - m_gradiente          
            h = (h + (1 + np.linalg.det(q.T * h * q)/np.linalg.det(p.T * q)) 
                 * (np.linalg.det(p * p.T)/np.linalg.det(p.T * q)) 
                 - np.linalg.det(p * q.T * h + h * q * p.T)/np.linalg.det(p.T * q))
            
            p_atual = p_prox
            cont_passos += 1

            if verbose:
                print(f"Passo Nº {cont_passos}, Ponto Atual = {p_atual}, "
                      f"Valor da F.O.: {self.calcular_func(self.func, self.vars, p_atual):.8f}\n")

        if verbose:
            print(f"Finalizado em {cont_passos} passos, com Ponto Ótimo = {p_atual} e "
                  f"Valor Ótimo da F.O.: {self.calcular_func(self.func, self.vars, p_atual):.8f}\n")

        return (p_atual, self.calcular_func(self.func, self.vars, p_atual), cont_passos, cont_armijo)
    
    def __repr__(self):
        return str(self.func)
    
if __name__ == '__main__':
    #Exemplo
    func = 'x**2 + (exp(x) - y)**2'
    vars = ['x', 'y']
    p_inicial = [0, 0]
    a = OtimizacaoNLP(func, vars, p_inicial)
    a.metodo_gradiente(verbose=True)
    #a.metodo_newton(verbose=True)
    #a.metodo_dfp(verbose=True)
    #a.metodo_bfgs(verbose=True)


    