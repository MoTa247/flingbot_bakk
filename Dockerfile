################################################################################
# BASE IMAGE: CUDA 9.2 + GL + Ubuntu 18.04
# -------------------------------------------------------
# Dieses alte Image ist zwingend für PyFlex, SoftGym und FlingBot erforderlich.
# Neuere CUDA-Versionen funktionieren NICHT mit PyFlex.
# Ubuntu 18.04 liefert GLIBC 2.27 → neuere Installer erfordern 2.28+.
################################################################################

# 12.02.2026 - Tanja Moser - Cuda/Ubuntu zu alt für VS Studio
FROM nvidia/cudagl:9.2-devel-ubuntu18.04
#FROM nvidia/cudagl:12.3.2-devel-ubuntu22.04

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
  && rm -rf /var/lib/apt/lists/*

################################################################################
# NVIDIA CAPABILITIES
################################################################################

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

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

# KORREKTE, GLIBC-KOMPATIBLE VERSION:
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh -O /tmp/miniconda.sh \
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
# UTF-8 SUPPORT (OPTIONAL, aber SEHR sinnvoll)
################################################################################

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

################################################################################
# DEFAULT COMMAND
################################################################################

CMD ["/bin/bash"]