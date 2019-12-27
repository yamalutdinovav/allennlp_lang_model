# Обучение моделей с использованием AllenNLP

Для корректной работы с зависимостями можно выполнить:
``` pip install -r requirements.txt ```

## Запуск программы

Для запуска программы в консоли необходисмо выполнить
``` python3 main.py [...] ```

Возможные аргументы:  
 `--model` — обучаемая модель (в настоящий момент доступны `lstm` и `transformer` 
 `--data-path` — путь к директории, в которой лежит датасет.  
 `--model-path` — путь к директории, в которую будут сохраняться обученные модели.  
 `--epochs` — число эпох обучения.  
 `--subwords` — если указан этот аргумент, обучение будет производиться на sub-words с использованием YouTokenToMe.  

Полный список аргументов и их функции:

`python3 main.py --help`
