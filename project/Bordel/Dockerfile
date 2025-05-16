#####################################
# RCP CaaS requirement (Image)
#####################################
# The best practice is to use an image
# with GPU support pre-built by Nvidia.
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/

# For example, if you want to use an image with pytorch already installed
# FROM nvcr.io/nvidia/pytorch:25.03-py3 or FROM nvcr.io/nvidia/ai-workbench/pytorch:1.0.6
# In this example, we'll use the second image.

FROM nvcr.io/nvidia/pytorch:25.03-py3

#####################################
# RCP CaaS requirement (Storage)
#####################################
# Create your user inside the container.
# This block is needed to correctly map
# your EPFL user id inside the container.
# Without this mapping, you are not able
# to access files from the external storage.
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

#####################################

# Copy your code inside the container
RUN mkdir -p /home/${LDAP_USERNAME}
COPY ./ /home/${LDAP_USERNAME}

# Set your user as owner of the new copied files
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

# Install required packages
RUN apt update
RUN apt install python3-pip -y

# Set the working directory in your user's home
WORKDIR /home/${LDAP_USERNAME}
USER ${LDAP_USERNAME}


# Install additional dependencies
RUN pip install -r requirements.txt

# Ajout de .local au PATH pour que pip sache où chercher les binaires
# ENV PATH=/home/${LDAP_USERNAME}/.local/bin:$PATH
# ENV PYTHONUSERBASE=/home/${LDAP_USERNAME}/.local

# # Installer pip et les paquets nécessaires dans l'environnement de l'utilisateur
# RUN python3 -m pip install --no-cache-dir --upgrade pip
# RUN python3 -m pip install --no-cache-dir --user -r requirements.txt

# # Assurer que le module est bien dans le bon chemin pour Python
# ENV PYTHONPATH=/home/${LDAP_USERNAME}/.local/lib/python3.8/site-packages:$PYTHONPATH
