"""
Client LLM (Azure OpenAI / OpenAI) — moteur de RAISONNEMENT de l'agent.

Le LLM pilote la CONVERSATION et l'EXTRACTION (comprendre un message libre, un
rapport de laboratoire océrisé, demander les valeurs manquantes). Il n'invente
JAMAIS de diagnostic : il appelle l'outil déterministe `assess_patient`
(modèle finetuned.keras + indice de Mentzer) qui reste la source de vérité.

Configuration par variables d'environnement (aucun secret dans le code) :
  Azure OpenAI (recommandé, cohérent avec Document Intelligence) :
    AZURE_OPENAI_ENDPOINT      ex. https://mon-ressource.openai.azure.com
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_DEPLOYMENT    ex. gpt-5.4   (nom du déploiement)
    AZURE_OPENAI_API_VERSION   ex. 2024-10-21   (optionnel)
  OpenAI direct (repli) :
    OPENAI_API_KEY
    OPENAI_MODEL               ex. gpt-5.4   (optionnel)

Si aucune clé n'est configurée, available() renvoie False et l'agent bascule
sur l'extraction hors-ligne (regex) — la démo reste reproductible sans réseau.
"""
from __future__ import annotations

import os


def available() -> bool:
    """Vrai si un backend LLM est configuré."""
    return bool(os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY")) \
        or bool(os.getenv("OPENAI_API_KEY"))


def get_chat_model(temperature: float = 0.1):
    """Retourne un modèle de chat LangChain (AzureChatOpenAI ou ChatOpenAI).

    Consommé par l'agent ReAct (care_agent) et le middleware de résumé.
    """
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            temperature=temperature,
        )
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5.4"),
                          temperature=temperature)
    raise RuntimeError(
        "Aucun backend LLM configuré (voir variables d'environnement).")
