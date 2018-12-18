

- Dans le calcul de la dérivée de G selon \theta pour la regression logistique bayesienne intervient
un produit matriciel n*n qui rend très difficile l'utilisation
de cet algorithme pour un nombre d'exemples conséquent

- Dans un premier temps on a cru que l'integrateur permettait d'obtenir des candidats pour
(theta, p) alors qu'il s'agit d'un ech de Gibbs.

- Auteur ne justifie pas pourquoi l'acceptance ratio doit être si élevé même si on comprend bien en
pratique qu'il faille un taux d'acceptation élevé