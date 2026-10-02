import json
import traceback

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django_ratelimit.decorators import ratelimit

from .services import repondre


MESSAGE_LONGUEUR_MAX = 500

def convertir_sources(sources):
    resultat = []

    for source in sources:
        if hasattr(source, "url"):
            try:
                resultat.append(source.url)
            except ValueError:
                resultat.append(str(source.name))
        else:
            resultat.append(str(source))

    return resultat

@csrf_exempt
@require_POST
@ratelimit(key="ip", rate="10/m", block=True)
def chat_api(request):

    #print("========== CHAT API ==========", flush=True)
    #print("BODY :", request.body, flush=True)

    try:
        donnees = json.loads(request.body)
        #print("JSON :", donnees, flush=True)

    except Exception as e:
        #print("ERREUR JSON :", repr(e), flush=True)
        traceback.print_exc()

        return JsonResponse(
            {"erreur": "Corps de requête JSON invalide."},
            status=400
        )

    message = donnees.get("message", "")

    #print("MESSAGE :", repr(message), flush=True)

    if not isinstance(message, str) or not message.strip():
        return JsonResponse(
            {
                "erreur": (
                    "Le champ 'message' est requis "
                    "et doit être une chaîne non vide."
                )
            },
            status=400
        )

    if len(message) > MESSAGE_LONGUEUR_MAX:
        return JsonResponse(
            {
                "erreur": (
                    f"Le message dépasse la longueur maximale "
                    f"de {MESSAGE_LONGUEUR_MAX} caractères."
                )
            },
            status=400
        )

    try:
        #print(">>> APPEL repondre()", flush=True)

        resultat = repondre(message)

        if "sources" in resultat:
            resultat["sources"] = convertir_sources(resultat["sources"])

        #print(">>> RESULTAT :", repr(resultat), flush=True)
        #print(">>> TYPE :", type(resultat), flush=True)

        return JsonResponse(resultat)

    except Exception as e:
        #print("========== ERREUR repondre() ==========", flush=True)
        #print("TYPE :", type(e).__name__, flush=True)
        #print("MESSAGE :", str(e), flush=True)
        #traceback.print_exc()

        return JsonResponse(
            {
                "erreur": "Une erreur interne est survenue.",
                "detail": str(e),
            },
            status=500
        )

def demo_page(request):
    return render(request, "chatbot/demo_page.html")
