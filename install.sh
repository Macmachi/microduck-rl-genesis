#!/usr/bin/env bash
# ============================================================================
#  INSTALLATION COMPLÈTE — Microduck RL sous Genesis
#
#  Pensé pour débutant : UNE commande, tout est isolé dans un conteneur,
#  RIEN n'est modifié sur ton système.
#
#      ./install.sh                 # détection automatique AMD / NVIDIA
#      ./install.sh --amd           # force la voie AMD (ROCm)     [TESTÉE]
#      ./install.sh --nvidia        # force la voie NVIDIA (CUDA)  [NON TESTÉE]
#
#  Ce que fait le script :
#    1. vérifie que distrobox + podman (ou docker) sont là ;
#    2. crée un conteneur depuis l'image officielle du fabricant de ta carte
#       (PyTorch GPU est DÉJÀ compilé dedans — c'est le plus dur de fait) ;
#    3. crée un environnement Python qui HÉRITE de ce PyTorch (surtout ne pas
#       le réinstaller par pip : on récupérerait une build CPU) ;
#    4. installe Genesis + les dépendances du projet, versions épinglées ;
#    5. vérifie que le GPU est bien vu de l'intérieur.
#
#  Réglable par variables d'environnement :
#      BOX_NAME=ma-box  BOX_IMAGE=...  VENV_DIR=...  ./install.sh
#
#  Désinstallation : ./uninstall.sh   (voir ./uninstall.sh --help)
# ============================================================================
set -e

VENDOR=""
for arg in "$@"; do
    case "$arg" in
        --amd|--rocm)    VENDOR="amd" ;;
        --nvidia|--cuda) VENDOR="nvidia" ;;
        -h|--help)       sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "Argument inconnu : $arg" >&2; exit 1 ;;
    esac
done

echo "== [1/6] Vérification des prérequis =="
if ! command -v distrobox >/dev/null 2>&1; then
    echo "ERREUR : distrobox n'est pas installé."
    echo "  Fedora :        sudo dnf install distrobox podman"
    echo "  Debian/Ubuntu : sudo apt install distrobox podman"
    echo "  Arch :          sudo pacman -S distrobox podman"
    exit 1
fi
if ! command -v podman >/dev/null 2>&1 && ! command -v docker >/dev/null 2>&1; then
    echo "ERREUR : ni podman ni docker n'est installé (voir ci-dessus)."
    exit 1
fi

# --- Détection de la carte ---------------------------------------------------
if [ -z "$VENDOR" ]; then
    if [ -e /dev/kfd ]; then
        VENDOR="amd"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        VENDOR="nvidia"
    else
        echo "ERREUR : aucun GPU détecté (/dev/kfd absent, nvidia-smi absent)."
        echo "         Force la voie voulue : ./install.sh --amd  ou  --nvidia"
        exit 1
    fi
fi

if [ "$VENDOR" = "amd" ]; then
    DEFAULT_IMAGE="docker.io/rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0"
    DEFAULT_BOX="genesis-box"
    GPU_FLAGS="--device /dev/kfd --device /dev/dri --group-add keep-groups"
    echo "   carte AMD détectée → voie ROCm (configuration testée)"
else
    DEFAULT_IMAGE="nvcr.io/nvidia/pytorch:24.10-py3"
    DEFAULT_BOX="genesis-box-cuda"
    GPU_FLAGS="--gpus all"
    cat <<'WARN'

   ############################################################
   #  VOIE NVIDIA : NON TESTÉE sur ce projet.                 #
   #  Le code ne dépend d'aucune API propre à AMD, donc ça     #
   #  devrait marcher, mais personne ne l'a vérifié. Il te     #
   #  faut le NVIDIA Container Toolkit installé côté hôte.     #
   #  Si ça casse, ouvre une issue avec la sortie de l'étape 6.#
   ############################################################

WARN
fi

BOX="${BOX_NAME:-$DEFAULT_BOX}"
IMAGE="${BOX_IMAGE:-$DEFAULT_IMAGE}"
VENV="${VENV_DIR:-}"   # vide = choisi après la création du conteneur
PROJET="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "   conteneur : $BOX"
echo "   image     : $IMAGE"
echo "   venv      : ${VENV:-auto (/opt/genesis/venv si présent, sinon ~/venvs/genesis)}"

echo "== [2/6] Création du conteneur (image ~15-20 Go au 1er téléchargement) =="
# Test d'existence par le moteur de conteneurs, en comparaison EXACTE (grep -qx).
# Pas `distrobox list | grep "$BOX"` : c'est une correspondance de sous-chaîne,
# donc chercher « genesis-box » trouverait « genesis-box-cuda » et réutiliserait
# le conteneur NVIDIA pour une installation AMD. Le format tabulaire de
# `distrobox list` peut en plus changer d'une version à l'autre ; `ps -a` non.
MOTEUR="$(command -v podman || command -v docker)"
if "$MOTEUR" ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$BOX"; then
    echo "   conteneur '$BOX' déjà présent — on le réutilise."
else
    distrobox create --name "$BOX" --image "$IMAGE" --yes \
        --additional-flags "$GPU_FLAGS"
fi

# Conteneur déjà équipé d'un venv commun (/opt/genesis/venv + lanceur
# `genesis`) : on s'en sert plutôt que d'en créer un second dans ~/venvs
# (~2,7 Go de doublon sur l'hôte). Le lanceur garde aussi les caches pip et
# Genesis dans le conteneur au lieu de ~/.cache.
LANCEUR=""
if [ -z "$VENV" ] && distrobox enter "$BOX" -- test -x /usr/local/bin/genesis >/dev/null 2>&1; then
    VENV=/opt/genesis/venv
    LANCEUR=/usr/local/bin/genesis
fi
VENV="${VENV:-$HOME/venvs/genesis}"

echo "== [3/6] Environnement Python ($VENV) =="
distrobox enter "$BOX" -- $LANCEUR bash -lc "
set -e
if [ ! -f '$VENV/bin/activate' ]; then
    python3 -m venv '$VENV'
    # On HÉRITE du PyTorch de l'image via un .pth — surtout PAS
    # --system-site-packages : un venv créé depuis /opt/venv remonterait au
    # python SYSTÈME (sans torch). On cherche où vit torch : /opt/venv (images
    # ROCm) ou le python système (la plupart des images CUDA).
    TORCH_SP=\$(/opt/venv/bin/python -c 'import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))' 2>/dev/null \
             || python3 -c 'import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))' 2>/dev/null || true)
    if [ -z \"\$TORCH_SP\" ]; then
        echo 'ERREUR : PyTorch introuvable dans cette image (ni /opt/venv, ni python système).'
        echo '         Prends une image qui embarque PyTorch (rocm/pytorch, nvcr.io/nvidia/pytorch, …).'
        exit 1
    fi
    VENV_SP=\$('$VENV/bin/python' -c 'import site; print(site.getsitepackages()[0])')
    echo \"\$TORCH_SP\" > \"\$VENV_SP/00_torch_inherit.pth\"
    echo \"   PyTorch hérité de : \$TORCH_SP\"
fi
source '$VENV/bin/activate'
pip install --quiet --upgrade pip

echo '== [4/6] Genesis + dépendances du projet =='
pip install --quiet -r '$PROJET/requirements.txt'

echo '== [5/6] Vérification GPU / versions =='
python - <<'PY'
import importlib.metadata as md
import torch, genesis
print('PyTorch  :', torch.__version__)
print('GPU vu   :', torch.cuda.is_available(),
      '(' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else '')
print('Genesis  :', genesis.__version__)
print('rsl-rl   :', md.version('rsl-rl-lib'))
PY

echo '== [6/6] Validation du portage (rapide) =='
cd '$PROJET'
python tests/test_external_torque.py
"

cat <<EOF

============================================================
 INSTALLATION TERMINÉE ✔
============================================================
 Entrer dans l'environnement :

   distrobox enter $BOX
   source $VENV/bin/activate
   cd "$PROJET"

 Puis :

   python tests/run_all.py                         # valider le portage
   python train.py -e smoke -B 64 --max-iterations 5   # smoke test
   python train.py -e microduck-velocity -B 4096   # entraînement complet
   tensorboard --logdir logs                       # suivre l'apprentissage

 Si "GPU vu : False" côté AMD : vérifie que ton utilisateur est dans
 les groupes video/render :
     sudo usermod -aG video,render \$USER
 puis déconnecte/reconnecte ta session.
============================================================
EOF
