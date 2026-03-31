################################################################################
# BASE IMAGE: CUDA 9.2 + GL + Ubuntu 18.04
# -------------------------------------------------------
# Dieses alte Image ist zwingend für PyFlex, SoftGym und FlingBot erforderlich.
# Neuere CUDA-Versionen funktionieren NICHT mit PyFlex.
# Ubuntu 18.04 liefert GLIBC 2.27 → neuere Installer erfordern 2.28+.
################################################################################

# 12.02.2026 - Tanja Moser - Cuda/Ubuntu zu alt für VS Studio
FROM nvidia/cudagl:9.2-devel-ubuntu18.04


################################################################################
# NICHT-INTERAKTIVE INSTALLATION (verhindert apt Dialoge)
################################################################################

ENV DEBIAN_FRONTEND=noninteractive

################################################################################
# SYSTEM-DEPENDENCIES
# -------------------------------------------------------
# glvnd, X11, OpenGL, build-essential usw. werden für PyFlex benötigt.
# bzip2 & ca-certificates werden für Miniconda benötigt.
################################################################################

RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
     libglvnd0 libgl1 libglx0 libegl1 libxext6 libx11-6 \
     cmake build-essential libgl1-mesa-dev freeglut3-dev libglfw3-dev libgles2-mesa-dev \
     openexr wget bzip2 ca-certificates curl \
     libopenexr-dev \
     #neues Image ab 1.44.2026
     libsdl2-2.0-0  \ # <-- benötigt für PyFlex Runtime (import pyflex)
     libilmbase-dev  # <-- benötigt für Python OpenEXR bindings
     # --- EGL FIX (AUSKOMMENTIERT FÜR TESTPHASE) ---
     libgles2 \
     libegl1-mesa-dev \
     libglvnd-dev \
     mesa-utils      # <-- enthält glxinfo (Debugging OpenGL/EGL)
  && rm -rf /var/lib/apt/lists/*

################################################################################
# NVIDIA CAPABILITIES
################################################################################

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
################################################################################
# EGL / HEADLESS RENDERING SETTINGS (für PyFlex)
# ------------------------------------------------------------------------------
# Wichtig für GPU-Rendering ohne Display (EGL)
# AKTUELL AUSKOMMENTIERT → erst aktivieren wenn getestet
################################################################################

# ENV NVIDIA_DRIVER_CAPABILITIES=all		#weniger stabil als Zeile 47
ENV PYOPENGL_PLATFORM=egl
ENV EGL_PLATFORM=surfaceless

WORKDIR /workspace

################################################################################
# INSTALL MINICONDA
# -------------------------------------------------------
# Ursprünglich wurde Miniconda "latest" installiert, aber:
#   → Der aktuelle Installer benötigt GLIBC ≥ 2.28
#   → Dieses Image hat GLIBC 2.27 → Installation würde abstürzen!
#
# Lösung: Wir verwenden Miniconda 4.9.2 (2020), die letzte Version kompatibel
# mit GLIBC 2.27 & CUDA 9.2.
################################################################################

ENV CONDA_DIR=/opt/conda

# Alte, inkompatible Version (NICHT MEHR BENUTZEN):
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

# KORREKTE, GLIBC-KOMPATIBLE VERSION:		"no check" im Fall von Netzwerkproblemen
RUN wget --quiet --no-check-certificate \
  https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh \
  -O /tmp/miniconda.sh \
  && /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR \
  && rm /tmp/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

################################################################################
# CONDA UPDATE / MAMBA INSTALLATION
# -------------------------------------------------------
# Ursprünglich vorgesehen:
#
#   RUN conda update -n base -c defaults conda -y
#   RUN conda install -n base -c conda-forge mamba -y
#
# Diese Schritte sind heute NICHT MEHR FUNKTIONAL:
#
# ❶ "conda update" versucht aktuelle Pakete zu installieren → benötigen GLIBC ≥ 2.28 → bricht ab.
# ❷ Der Solver von conda-forge ist 2025 enorm groß → benötigt 4–6 GB RAM.
#    Docker-Build-Umgebungen haben nur ~2 GB RAM → führt zu OOM („cannot allocate memory“).
# ❸ Der mamba-Installer benötigt ebenfalls moderne Abhängigkeiten → ebenfalls unbrauchbar.
#
# DESHALB DARF CONDA HIER NICHT MEHR AKTUALISIERT WERDEN!
################################################################################

# ORIGINAL (Fehlerhaft, aber zur Dokumentation belassen):
# RUN conda update -n base -c defaults conda -y \
#   && conda install -n base -c conda-forge mamba -y

# KORREKTE LÖSUNG:
RUN echo "Conda remains at base version (4.9.2) – no update performed due to GLIBC & memory limits"



################################################################################
# OPENEXR (für SoftGym / PyFlex Rendering)              wsl jetzt nicht nötig/möglich wegen glib>=2.28
# ------------------------------------------------------------------------------
# Wird über conda installiert, damit IlmBase / ABI korrekt aufeinander abgestimmt sind.
# Versionen sind gepinnt → reproduzierbarer Docker-Build ohne Solver-Probleme.
################################################################################

################################################################################
#OpenEXR könnte im Dockerfile später zu Problemen führen, da Updates erfolgt sind:
#conda -> 22.9.0     libstdcxx/libgcc -> neuere Toolchain   openexr ->3.1.11
################################################################################

# RUN conda install -y -c conda-forge \
#     openexr=3.1.11 \
#     openexr-python=1.3.9 \
#     imath=3.1.9
################################################################################
# ALIASING VON CONDA → MAMBA
# -------------------------------------------------------
# Ursprünglich als Qualitätsverbesserung gedacht:
#
#   RUN echo \"alias conda='mamba'\" >> ...
#
# Problem:
#   → mamba ist NICHT installiert
#   → würde conda unbenutzbar machen
################################################################################

# ORIGINAL (auskommentiert, NICHT verwenden):
# RUN echo \"alias conda='mamba'\" >> $CONDA_DIR/etc/profile.d/conda.sh

################################################################################
# CONDA IN BASH AUTOMATISCH LADEN
# -------------------------------------------------------
# Dies ist korrekt und notwendig.
################################################################################

RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> /etc/bash.bashrc

################################################################################
# PYFLEX ENV VARS FÜR INTERAKTIVE SHELLS
################################################################################

RUN echo "export PYFLEXROOT=/workspace/PyFlex/" >> /etc/bash.bashrc && \
    echo "export PYTHONPATH=/workspace/PyFlex/bindings/build:\$PYTHONPATH" >> /etc/bash.bashrc

################################################################################
# UTF-8 SUPPORT (OPTIONAL, aber SEHR sinnvoll)
################################################################################

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

#####Hinzugefügt#####

################################################################################
# PYTHON / TORCH INSTALLATION
# ------------------------------------------------------------------------------
# Wir benutzen pip innerhalb der Base-Environment.
# Torch 1.4 ist die letzte stabile CUDA 9.2 kompatible Version.
################################################################################


#RUN pip install --upgrade "pip<23" setuptools wheel	#notfalls um stabilität mit Python3.7 zu gewährleisten
RUN pip install --upgrade pip setuptools wheel

# Torch MUSS VOR requirements installiert werden
RUN pip install torch==1.4.0 torchvision==0.5.0
#RUN pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 \	#notfalls ersatz um CUDA9.2 zu sichern
#    -f https://download.pytorch.org/whl/torch_stable.html
# wenn ich torch und cuda hier anpasse muss ich in requirements.txt ebenfalls +cu92 bei beiden hinzufügen! sonst mögliche kollision


################################################################################
# REQUIREMENTS INSTALLIEREN (Docker Cache optimiert)
# ------------------------------------------------------------------------------
# Nur requirements.txt kopieren → schneller Rebuild bei Codeänderungen
################################################################################

COPY requirements.txt .

RUN pip install -r requirements.txt


################################################################################
# RESTLICHES PROJEKT KOPIEREN
################################################################################

COPY . .

################################################################################
# PYFLEX ENVIRONMENT VARIABLES			[31.3.2026]
# ------------------------------------------------------------------------------
# Wichtig für PyFlex bindings (Python findet pyflex.so)
################################################################################

ENV PYFLEXROOT=/workspace/PyFlex/
ENV PYTHONPATH=/workspace/PyFlex/bindings/build:${PYTHONPATH}

################################################################################
# Python Logging sofort anzeigen (kein Buffering)
################################################################################

ENV PYTHONUNBUFFERED=1


####bis hier    25.02.2026####
#### info von 31.3.2026####
# NOTE:
# Diese EGL Settings sind notwendig, damit PyFlex im Docker (headless GPU)
# korrekt läuft. Ohne diese kommt es zu:
#   eglInitialize() failed
#   SIGSEGV in SimEnv
###########################

################################################################################
# DEFAULT COMMAND
################################################################################

CMD ["/bin/bash"]
