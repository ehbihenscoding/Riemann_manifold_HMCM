

- Dans le calcul de la dérivée de G selon \theta pour la regression logistique bayesienne intervient
un produit matriciel n*n qui rend très difficile l'utilisation
de cet algorithme pour un nombre d'exemples conséquent

- Dans un premier temps on a cru que l'integrateur permettait d'obtenir des candidats pour
(theta, p) alors qu'il s'agit d'un ech de Gibbs.
Problème a subsisté mais maintenant semble ok!

- Auteur ne justifie pas pourquoi l'acceptance ratio doit être si élevé même si on comprend bien en
pratique qu'il faille un taux d'acceptation élevé.
En fait logique: plus l'erreur réalisée par l'intégrateur est élevée, 
moins on a de chances d'accepter un échantillon
Pas clair le choix de l'acceptance ratio optimal

- Observation: quand initialisation très éloignée des modes de la fonction,
remarque que Vanilla reste potentiellement bloqué alors que RHMC non!
Pour ca, regarder 1D, partir de theta_0 = 100
Décorrelation très rapide

En théorie: on a acceptance rates optimaux pour HMC classique
https://aip.scitation.org/doi/abs/10.1063/1.3498436

En pratique: pas évident de tuner eps afin d'avoir un acceptance rate donné
En pratique: ces articles font le choix d'acceptance rates très élevés
https://arxiv.org/pdf/1407.1517.pdf
https://arxiv.org/pdf/1212.4693.pdf

