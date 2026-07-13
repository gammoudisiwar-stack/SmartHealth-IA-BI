"""
Azure AI Document Intelligence — océrisation des rapports de laboratoire.

Permet à l'utilisateur de TÉLÉVERSER un compte-rendu NFS (PDF/image) ; le service
extrait le texte (et les paires clé/valeur), puis le LLM (llm.extract_fields)
structure les valeurs biologiques pour l'agent.

Configuration par variables d'environnement (aucun secret dans le code) :
  AZURE_DOC_INTEL_ENDPOINT   ex. https://mon-di.cognitiveservices.azure.com
  AZURE_DOC_INTEL_KEY

Si non configuré, available() renvoie False (l'UI proposera la saisie manuelle).
"""
from __future__ import annotations

import os

_CLIENT = None


def available() -> bool:
    return bool(os.getenv("AZURE_DOC_INTEL_ENDPOINT") and os.getenv("AZURE_DOC_INTEL_KEY"))


def get_client():
    global _CLIENT
    if _CLIENT is None:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
        _CLIENT = DocumentIntelligenceClient(
            endpoint=os.environ["AZURE_DOC_INTEL_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_DOC_INTEL_KEY"]),
        )
    return _CLIENT


def ocr_text(file_bytes: bytes) -> str:
    """Océrise un document (PDF/image) et renvoie du MARKDOWN structuré.

    Utilise le modèle 'prebuilt-layout' avec sortie Markdown : il préserve la
    structure (titres, TABLEAUX de résultats NFS, paires clé/valeur), ce qui
    aide fortement le LLM à extraire les bonnes valeurs.
    """
    from azure.ai.documentintelligence.models import (
        AnalyzeDocumentRequest, DocumentContentFormat)
    client = get_client()
    poller = client.begin_analyze_document(
        "prebuilt-layout",
        AnalyzeDocumentRequest(bytes_source=file_bytes),
        output_content_format=DocumentContentFormat.MARKDOWN,
    )
    result = poller.result()
    if getattr(result, "content", None):
        return result.content
    # Repli : reconstituer le texte ligne par ligne.
    lines = []
    for page in getattr(result, "pages", []) or []:
        for line in getattr(page, "lines", []) or []:
            lines.append(line.content)
    return "\n".join(lines)
