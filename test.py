class mojaLista(list):
    def deleteAll(self, a):
        while a in self:
            self.remove(a)



lista = mojaLista(['a','b','a','a','a'])
lista.deleteAll('a')


print(lista)