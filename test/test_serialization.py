import mtj_softtuner


def test_serialization():
    universes = (1, 0, 1729, -1729, None)
    trainers = [mtj_softtuner.BasicTrainer(u) for u in universes]
    params = ({"a": 1}, {"4": "2"}, {"c": None}, {"d": [1, 2]}, {"e": "e"})

    for t, p in zip(trainers, params):
        t.data.params = p
    for t in trainers:
        t.save_data()

    trainers = [mtj_softtuner.BasicTrainer(u) for u in universes]

    for u, t, p in zip(universes, trainers, params):
        if u is not None:
            assert t.data.params == p
        else:
            assert t.data.params is None
