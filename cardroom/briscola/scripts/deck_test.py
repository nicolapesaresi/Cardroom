from cardroom.briscola.game.dealer import BriscolaDealer

if __name__=="__main__":
    dealer = BriscolaDealer()
    for card in dealer.deck:
        print(card)
        print(card.card_id)
    print(len(dealer.deck))