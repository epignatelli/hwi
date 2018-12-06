

def get_triples(data_gen, Xa, Xn, Xp, Y):

        genXa = data_gen.flow(Xa, Y)
        genXn = data_gen.flow(Xn)
        genXp = data_gen.flow(Xp)

        while True:
            Xa_i = genXa.next()
            Xn_i = genXn.next()
            Xp_i = genXp.next()
            yield (Xa_i[0], Xn_i, Xp_i), Xa_i[1]