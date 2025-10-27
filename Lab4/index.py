import numpy as np
import matplotlib.pyplot as plt

# Krok 1: Definicja funkcji celu i jej gradientu
def f(x, y):
    """Funkcja celu: f(x, y) = |x - y| + x^2"""
    return np.abs(x - y) + x**2

def grad_f(x, y):
    """Gradient funkcji celu."""
    grad_x = np.sign(x - y) + 2 * x
    grad_y = -np.sign(x - y)
    return np.array([grad_x, grad_y])

# Krok 2: Adaptacja funkcji optymalizującej z instrukcji
def optimize_path(opt_name, learning_rate, steps=500, start_pos=np.array([2.0, -2.0])):
    """Uruchamia wybrany algorytm optymalizacji."""
    pos = start_pos.copy()
    path = [pos.copy()]
    
    v = np.zeros(2)
    s = np.zeros(2)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    
    for t in range(1, steps + 1):
        grad = grad_f(*pos)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1.0:
            grad /= grad_norm
        
        if opt_name == 'gd':
            pos -= learning_rate * grad
        elif opt_name == 'momentum':
            v = beta1 * v + learning_rate * grad
            pos -= v
        elif opt_name == 'rmsprop':
            s = beta2 * s + (1 - beta2) * grad**2
            pos -= learning_rate * grad / (np.sqrt(s) + eps)
        elif opt_name == 'adam':
            v = beta1 * v + (1 - beta1) * grad
            s = beta2 * s + (1 - beta2) * grad**2
            v_corr = v / (1 - beta1**t)
            s_corr = s / (1 - beta2**t)
            pos -= learning_rate * v_corr / (np.sqrt(s_corr) + eps)
            
        path.append(pos.copy())
        
    return np.array(path), steps  

# Krok 3: Wizualizacja i analiza wyników dla różnych współczynników uczenia
learning_rates = [0.1, 0.05, 0.005]
optimizers = ['gd', 'momentum', 'rmsprop', 'adam']
colors = ['blue', 'green', 'red', 'purple']  
start_points = [np.array([2.0, -2.0]), np.array([1.0, 1.0])]  

x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

for start_pos in start_points:
    print(f"\nAnaliza dla punktu startowego: {start_pos}")
    for lr in learning_rates:
        plt.figure(figsize=(12, 8))
        plt.contour(X, Y, Z, levels=np.logspace(0, 3, 20), cmap='gray')
        
        for i, opt_name in enumerate(optimizers):
            path, steps = optimize_path(opt_name, learning_rate=lr, start_pos=start_pos)  
            plt.plot(path[:, 0], path[:, 1], label=opt_name.upper(), color=colors[i], linestyle='--')
            
            final_val = f(path[-1, 0], path[-1, 1])
            conv_iter = next((i for i, p in enumerate(path) if f(p[0], p[1]) < 0.01), steps)  
            print(f"{opt_name.upper()} (η={lr}): Końcowa wartość f = {final_val:.4f}, Konwergencja po {conv_iter} iteracjach")
        
        plt.title(f'Trajektorie optymalizacji dla η = {lr} (start: {start_pos})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()
