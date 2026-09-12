"""Demo knowledge base for Tunisian French ISP call-center testing.

Used by:
- POST /api/v1/tenants/{id}/demo-knowledge
- scripts/make_sample_pdf.py
- scripts/seed_demo_rag.py
"""

from __future__ import annotations

from app.models.schemas import PageText

DEMO_DOCUMENT_NAME = "FAQ_Orange_TN_Demo_v2.pdf"

# Each entry becomes one indexed page (chunked further if long).
DEMO_PAGES: list[str] = [
    """FAQ Support Client — Opérateur Internet Tunisie (démo Vantage AI)
Document de procédures pour le centre d'appels francophone.
Langue: français. Marché: Tunisie. Devise: dinar tunisien (DT).
Numéro d'assistance: 1000. Horaires: 8h–20h, 7j/7.
Escalade humaine: transfert vers un conseiller senior après 2 diagnostics échoués.
Règle agent: pour toute procédure technique multi-étapes, guider le client étape par étape
et vérifier avec lui après chaque étape avant de continuer.""",
    """Panne de connexion Internet — diagnostic niveau 1 (guidé)
Conduire le diagnostic une étape à la fois. Après chaque étape, demander: « Dites-moi quand c'est fait » ou « Est-ce que ça a marché ? »
Étape A: Éteindre le routeur 2 minutes, le rallumer, attendre 3 minutes. Vérifier avec le client.
Étape B: Vérifier les câbles RJ45 entre le routeur et l'ordinateur ou la box TV. Vérifier avec le client.
Étape C: Tester sur un autre appareil (smartphone en Wi-Fi). Vérifier avec le client.
Étape D: Vérifier les voyants: Power fixe, Fibre/DSL fixe, Wi-Fi allumé. Demander ce que le client voit.
Si le voyant Fibre/DSL reste rouge après ces étapes: ouvrir un ticket panne réseau, délai moyen 4 heures ouvrables.""",
    """Wi-Fi lent ou instable
Causes fréquentes: trop d'appareils, mur épais, canal saturé, ancien routeur.
Actions conseillées (une à la fois, avec confirmation client):
- Se placer à moins de 5 mètres du routeur — vérifier.
- Redémarrer le routeur — vérifier.
- Séparer les réseaux 2.4 GHz et 5 GHz si disponible — vérifier.
Mot de passe Wi-Fi: accessible dans l'espace client ou sur l'étiquette du routeur.
Débit minimum garanti Fibre 100 Mbps: 70 Mbps en conditions normales hors pic.
Si le débit reste sous 20 Mbps après diagnostic: planifier un technicien domicile (délai 48–72h).""",
    """Réinitialisation du routeur — procédure guidée (vérification après chaque étape)
IMPORTANT pour l'agent: ne jamais donner toutes les étapes d'un coup.
Donner uniquement l'étape en cours, puis attendre la confirmation du client (« oui », « c'est fait », « OK ») avant l'étape suivante.

Avant de commencer: prévenir que la réinitialisation efface les réglages Wi-Fi personnalisés.
Le technicien à domicile n'est pas obligatoire pour un simple reset.

Étape 1: Demander au client de localiser le bouton Reset (petit trou à l'arrière du boîtier, souvent avec une trombone).
Demander: « Vous voyez le bouton Reset ? Dites-moi quand vous l'avez trouvé. »

Étape 2 (après confirmation): Demander de maintenir Reset pendant 10 secondes jusqu'au clignotement rouge.
Demander: « Le voyant a clignoté en rouge ? »

Étape 3 (après confirmation): Demander d'attendre le redémarrage complet, environ 5 minutes, sans toucher au boîtier.
Demander: « Les voyants sont-ils redevenus stables ? »

Étape 4 (après confirmation): Se reconnecter avec le SSID et le mot de passe d'usine indiqués sous le boîtier.
Demander: « Êtes-vous reconnecté au Wi-Fi ? Internet fonctionne-t-il maintenant ? »

Si Internet ne revient pas après l'étape 4: ouvrir un ticket technique niveau 2.""",
    """Forfaits Fibre — tarifs 2026
Fibre 50 Mbps: 39 DT / mois. Frais de mise en service: 30 DT.
Fibre 100 Mbps (aussi dit 100 mégabits, 100 Mb/s, cent méga, forfait 100): 59 DT / mois. Frais de mise en service: 50 DT.
Fibre 300 Mbps: 89 DT / mois. Frais de mise en service: 50 DT.
Engagement standard: 12 mois. Sans engagement: +10 DT / mois.
Promotion en cours (mars–juin): Fibre 100 à 49 DT les 3 premiers mois pour les nouveaux clients.
Les tarifs incluent la box Wi-Fi. Le décodeur TV est optionnel (+8 DT / mois)."""
    """Facturation et paiements
La facture est émise le 1er de chaque mois. Échéance: le 15.
Modes de paiement: agence, espace client, carte bancaire, Flouci, charge électronique.
Retard de paiement: rappel SMS J+3, restriction débit J+10, suspension J+20.
Réactivation après paiement: sous 2 heures ouvrables.
Réclamation facture: ouvrir un ticket « facturation » avec le numéro de ligne et le mois contesté.
Délai de traitement d'une réclamation facture: 5 jours ouvrables.""",
    """Déménagement et transfert de ligne
Le client peut demander le transfert de sa ligne Fibre vers une nouvelle adresse en Tunisie.
Conditions: couverture Fibre disponible à la nouvelle adresse (vérifier code postal / gouvernorat).
Délai standard: 7 à 10 jours ouvrables après validation technique.
Frais de transfert: 40 DT. Pas de nouveau engagement si l'ancien contrat est encore actif.
Documents: CIN + preuve de domicile (facture STEG ou contrat de bail).
Pendant le transfert, l'ancienne adresse est suspendue le jour de l'activation de la nouvelle.""",
    """Changement de forfait et résiliation
Upgrade (ex: 50 → 100 Mbps): immédiat, facturation au prorata.
Downgrade: possible à la date d'anniversaire mensuelle.
Résiliation: préavis 30 jours. Pénalité de rupture anticipée: 3 mois de forfait restants (engagement).
Retour du matériel: box + alimentation en agence sous 15 jours, sinon facturation 180 DT.
Le numéro 1000 peut enregistrer une demande de résiliation; confirmation écrite par SMS.""",
    """Sécurité et phishing
Ne jamais demander le mot de passe espace client, le code OTP SMS, ou le PIN carte bancaire.
Si le client reçoit un appel suspect se présentant comme l'opérateur: lui dire de raccrocher et de rappeler le 1000.
Changement de mot de passe espace client: menu Sécurité → Mot de passe → lien email.
SIM / eSIM: hors périmètre Fibre; rediriger vers le service mobile si la question concerne le mobile.""",
    """Escalade et SLA
Niveau 1 (agent IA / conseiller): diagnostic standard, tarifs, procédures guidées étape par étape.
Niveau 2: panne confirmée, tickets techniques, litiges facture > 20 DT.
Niveau 3: incident massif régional, décision commerciale exceptionnelle.
SLA panne individuelle: rétablissement sous 4 heures ouvrables (cible 80%).
SLA incident massif: communication client sous 30 minutes, MAJ toutes les heures.
Toujours citer la source documentaire (page FAQ) quand une politique est évoquée.""",
]


def demo_page_texts() -> list[PageText]:
    return [PageText(page=i, text=text.strip()) for i, text in enumerate(DEMO_PAGES, start=1)]


def demo_test_questions() -> list[str]:
    """Suggested voice/text questions for manual QA."""
    return [
        "Ma connexion internet ne fonctionne plus, que dois-je faire ?",
        "Combien coûte le forfait Fibre 100 ?",
        "Comment réinitialiser mon routeur ?",
        "Quels sont les délais en cas de panne réseau ?",
        "Comment payer ma facture et que se passe-t-il en cas de retard ?",
        "Je déménage à Sousse, comment transférer ma ligne ?",
        "Je veux résilier mon abonnement, quelle est la procédure ?",
    ]
