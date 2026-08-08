from django.shortcuts import render
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django_ratelimit.decorators import ratelimit
from .services import repondre

MESSAGE_LONGUEUR_MAX = 500


@csrf_exempt
@require_POST
@ratelimit(key="ip", rate="10/m", block=True)
def chat_api(request):
    try:
        donnees = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"erreur": "Corps de requete JSON invalide."}, status=400)

    message = donnees.get("message", "")

    if not isinstance(message, str) or not message.strip():
        return JsonResponse({"erreur": "Le champ \'message\' est requis et doit etre une chaine non vide."}, status=400)

    if len(message) > MESSAGE_LONGUEUR_MAX:
        return JsonResponse(
            {"erreur": f"Le message depasse la longueur maximale de {MESSAGE_LONGUEUR_MAX} caracteres."},
            status=400,
        )

    resultat = repondre(message)
    return JsonResponse(resultat)


def demo_page(request):
    return render(request, "chatbot/demo_page.html")
