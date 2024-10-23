# Lab3ASIs26099
Implementacja zadania nr 3 z ASI
Generalnie projekt składa sie z 2 głównych plików zawartych w pliku asi3Project 
który był stworzony na potrzeby modyfikacji lokalnej kodu w pycharm i mozliwosci 
szybkiego pushowania zmian z poziomu pycharm

czyszczenie_porównywanieModeli.py -
Plik wykonujacy wszelkie operacje przygotowywujace dane, wypełniający braki, pobiera dane i 
tranuje przykladowe modele, wybiera najlepszy który potem trafia do pliku trening i raport

trening_i_raport.py - generalnie ten plik importuje i wtedy gdy model jest niedostateczny wykonywana jest optymalizacja na 2 sposoby
 tunowanie hiperparametrów, walidacja krzyżowa. Po tym wszystkim pokazuje po prostu jakie poprawy zaszły po optymalizacji kodu.

Dokumentacja jest wynikiem automatycznie tworzonej dokumentacji która pisana jest w trakcie kompilowania kodu python.
Starałem sie zrobic dynamiczna analize i pliki te mozna znalezc w wykonanym workflow.
Są rodzielone na 2 poniewaz te dokumentacje dotycza dwoch róznych rzeczy plik z nazwa analiza jest głównym plikiem z
dokumentacja na stepny dotyczy jednie jakie zostały wykonane kroki optymalizacyjne ( generalnie kroki ktore zostały wykonane w trening )
 
