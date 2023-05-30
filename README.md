# BIU_Ex2

##How to run the code?

it's better to run the code inside a venv or conda environment

### install packages 
```
pip3 install -r requirements.txt
```

### Run the file and add parameters

####The file can get two parameters
1. The first parameter is the path to the encrypted file (mandatory)
1. The first parameter is a sovler type - can be one of three - regular, lamark, darwin (optional, default - regular)
1. The second parameter is a boolean paramter which determines if to verbose display in each iteration the decrypted text
   (optional, default-true)

```
python3 main.py data/enc.txt regular true
```


For cloning the project  

```
git clone https://github.com/itaimondshine/BIU_Ex2.git
```
