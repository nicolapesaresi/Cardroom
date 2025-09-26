from cardroom.briscola.game.dealer import BriscolaDealer

if __name__=="__main__":
    dealer = BriscolaDealer()
    for card in dealer.deck:
        print(card)
    print(len(dealer.deck))