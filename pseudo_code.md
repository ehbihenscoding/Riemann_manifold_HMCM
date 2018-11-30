
- faire une classe pour les priors L(\theta)
de sorte à ce que l'on ait acces facilement à leur dérivée 
et à l'expression sous forme analytique



- faire une classe pour G
son expression selon paramètre
son inverse

- faire une classe p et theta qui géreront les updates du leapfrog integrator
et qui sauront gérer les cas où G est constante (càd où les expressions sont directement modifiées) et les cas où
G n'est pas constante et où un algo de point fixe permet de faire l'update.


- Sampling procedure:

inputs: p_0, \theta_0, G_0:

fonctions: grad_\theta_H
grad_p_H
G_inv
    
    sample = []
    for i=0:N-1
       sample p_i+1 selon N(0, G_i)
        
        
       !Important: regarder influences des nb iter
       for j=1:Nb_leapfrogs:
           p_eps/2 = p_i+1
           for k_pt_fixe=1:n_pt_fixe:
             p_eps/2 = p_i+1 - eps/2*grad_\theta_H(theta_i, p_eps/2)
           endfor
           
           theta_eps = theta_i
           for k_pt_fixe=1:n_pt_fixe:
             theta_eps = theta_i + eps/2*[G_inv(theta_i) + G_inv(theta_eps)]p
           
           p_i+1 = p_eps/2 - eps/2*grad_\theta_H(theta_eps, p_eps/2)      
    
        ajouter theta_eps a sample
        
        
        
calcul de grad_\theta_H
Assumption \grad log(|X|) = X_inv

grad_\theta_H = -grad L + 1/2*1/(2pi)*G_inv(\theta)

- Inverse of theta computation: