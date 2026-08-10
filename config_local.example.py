# Modele de configuration locale : copier vers config_local.py et renseigner
# les valeurs reelles. config_local.py est ignore par git (secrets).

# URL du flux JPEG de la camera IP
CAMERA_URL = "http://192.168.1.100/cgi-bin/viewer/video.jpg?streamid=3&resolution=640x480&quality=5"

# Identifiants HTTP de la camera (utilisateur, mot de passe)
CAMERA_AUTH = ("user", "password")

# Topic ntfy prive pour les notifications avec image
NTFY_TOPIC = "https://ntfy.sh/your-private-topic"
