import subprocess


def create_conda_env_sphere():
    print("Creating Conda environment env_sphere")
    subprocess.check_call(["conda", "create", "--name", "env_sphere", "--yes", "python=3.7.9"])


def create_conda_env_randlanet():
    print("Creating Conda environment env_randlanet")
    subprocess.check_call(["conda", "create", "--name", "env_randlanet", "--yes", "python=3.8.18"])


def install_dependencies_sphere():
    print("Installing dependencies from requirements_sphere.txt in Conda environment env_sphere")
    # Activate the Conda environment
    subprocess.check_call(
        ["conda", "run", "-n", "env_sphere", "pip", "install", "torch==1.8.0+cu111", "-f",
         "https://download.pytorch.org/whl/torch_stable.html"])

    subprocess.check_call(
        ["conda", "run", "-n", "env_sphere", "pip", "install", "torchvision==0.9.0+cu111", "-f",
         "https://download.pytorch.org/whl/torch_stable.html"])

    subprocess.check_call(
        ["conda", "run", "-n", "env_sphere", "pip", "install", "torchaudio==0.8.0", "-f",
         "https://download.pytorch.org/whl/torch_stable.html"])

    subprocess.check_call(
        ["conda", "run", "-n", "env_sphere", "pip", "install", "torch-sparse==0.6.11", "-f",
         "https://download.pytorch.org/whl/torch_stable.html"])

    subprocess.check_call(
        ["conda", "run", "-n", "env_sphere", "pip", "install", "torch-cluster==1.6.0"])

    subprocess.check_call(
        ["conda", "run", "-n", "env_sphere", "pip", "install", "torch_scatter==2.0.9"])
    subprocess.check_call(
        ["conda", "run", "-n", "env_sphere", "pip", "install", "torch_geometric==1.7.2"])
    subprocess.check_call(
        ["conda", "run", "-n", "env_sphere", "pip", "install", "-r", "requirements_sphere.txt"])
    subprocess.run(
        ["conda", "run", "-n", "env_sphere", "conda", "install", "-c", "conda-forge", "libstdcxx-ng"], input='y\n',
        text=True)
    print("Finished installing dependencies")

def install_dependencies_randlanet():
    print("Installing dependencies from requirements_randlanet.txt in Conda environment env_randlanet")
    # Activate the Conda environment
    subprocess.check_call(
        ["conda", "run", "-n", "env_randlanet", "pip", "install", "-r", "requirements_randlanet.txt"])
    subprocess.run(
        ["conda", "run", "-n", "env_randlanet", "conda", "install", "-c", "conda-forge", "libstdcxx-ng"], input='y\n',
        text=True)
    print("Finished installing dependencies")

if __name__ == "__main__":
    create_conda_env_sphere()
    install_dependencies_sphere()
    create_conda_env_randlanet()
    install_dependencies_randlanet()
