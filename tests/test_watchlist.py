def test_watchlist_requires_auth(client):
    assert client.get("/watchlist").status_code == 401


def test_add_unknown_symbol_404(client, auth_headers):
    resp = client.post("/watchlist", json={"symbol": "NOPE"}, headers=auth_headers)
    assert resp.status_code == 404


def test_add_and_list_symbol(client, seeded, auth_headers):
    resp = client.post("/watchlist", json={"symbol": "nabil"}, headers=auth_headers)
    assert resp.status_code == 201
    assert resp.json() == {"symbol": "NABIL"}
    assert client.get("/watchlist", headers=auth_headers).json() == [{"symbol": "NABIL"}]


def test_add_duplicate_conflicts(client, seeded, auth_headers):
    client.post("/watchlist", json={"symbol": "NABIL"}, headers=auth_headers)
    resp = client.post("/watchlist", json={"symbol": "NABIL"}, headers=auth_headers)
    assert resp.status_code == 409


def test_delete_symbol(client, seeded, auth_headers):
    client.post("/watchlist", json={"symbol": "NABIL"}, headers=auth_headers)
    assert client.delete("/watchlist/NABIL", headers=auth_headers).status_code == 204
    assert client.get("/watchlist", headers=auth_headers).json() == []


def test_delete_missing_404(client, auth_headers):
    assert client.delete("/watchlist/NABIL", headers=auth_headers).status_code == 404


def test_users_cannot_see_each_others_lists(client, seeded, make_headers):
    alice = make_headers("alice@b.com")
    bob = make_headers("bob@b.com")
    client.post("/watchlist", json={"symbol": "NABIL"}, headers=alice)
    assert client.get("/watchlist", headers=bob).json() == []
    assert client.delete("/watchlist/NABIL", headers=bob).status_code == 404
