import os
import gfwapiclient as gfw

def get_gfw_client(token: str = None) -> gfw.Client:
    """
    Crée et retourne une instance du client Global Fishing Watch.
    Cherche le token dans la variable d'environnement GFW_API_ACCESS_TOKEN.
    """
    token = token or os.getenv("GFW_API_ACCESS_TOKEN")
    if not token:
        raise ValueError("Merci de définir GFW_API_ACCESS_TOKEN dans votre environnement.")
    return gfw.Client(access_token=token)

# instance singleton à réutiliser
gfw_client: gfw.Client = get_gfw_client()
