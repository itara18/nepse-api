def test_register_creates_user(client):
    resp = client.post(
        "/auth/register", json={"email": "a@b.com", "password": "password123"}
    )
    assert resp.status_code == 201
    assert resp.json()["email"] == "a@b.com"
    assert "password" not in resp.json()


def test_register_duplicate_email_conflicts(client):
    body = {"email": "a@b.com", "password": "password123"}
    client.post("/auth/register", json=body)
    assert client.post("/auth/register", json=body).status_code == 409


def test_register_rejects_short_password(client):
    resp = client.post("/auth/register", json={"email": "a@b.com", "password": "short"})
    assert resp.status_code == 422


def test_login_wrong_password_rejected(client):
    client.post("/auth/register", json={"email": "a@b.com", "password": "password123"})
    resp = client.post("/auth/login", data={"username": "a@b.com", "password": "nope"})
    assert resp.status_code == 401


def test_me_requires_token(client):
    assert client.get("/auth/me").status_code == 401


def test_me_rejects_garbage_token(client):
    resp = client.get("/auth/me", headers={"Authorization": "Bearer garbage"})
    assert resp.status_code == 401


def test_me_returns_current_user(client, auth_headers):
    resp = client.get("/auth/me", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["email"] == "a@b.com"


def test_login_is_rate_limited(client):
    codes = [
        client.post("/auth/login", data={"username": "x@y.com", "password": "wrongpass1"}).status_code
        for _ in range(12)
    ]
    assert codes[-1] == 429
    assert 429 not in codes[:10]
