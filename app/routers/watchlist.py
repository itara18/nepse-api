from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import SymbolPath, get_current_user
from app.models import Price, User, WatchlistItem
from app.schemas import WatchlistIn, WatchlistOut

router = APIRouter(prefix="/watchlist", tags=["watchlist"])


def _find(db: Session, user: User, symbol: str) -> WatchlistItem | None:
    return db.scalar(
        select(WatchlistItem).where(
            WatchlistItem.user_id == user.id, WatchlistItem.symbol == symbol
        )
    )


@router.get("", response_model=list[WatchlistOut])
def list_items(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.scalars(
        select(WatchlistItem)
        .where(WatchlistItem.user_id == user.id)
        .order_by(WatchlistItem.symbol)
    ).all()


@router.post("", response_model=WatchlistOut, status_code=201)
def add_item(
    payload: WatchlistIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    symbol = payload.symbol.upper()
    if db.scalar(select(Price.id).where(Price.symbol == symbol).limit(1)) is None:
        raise HTTPException(status_code=404, detail=f"Unknown symbol {symbol}")
    if _find(db, user, symbol):
        raise HTTPException(status_code=409, detail="Already in watchlist")
    item = WatchlistItem(user_id=user.id, symbol=symbol)
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@router.delete("/{symbol}", status_code=204)
def remove_item(
    symbol: SymbolPath,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    item = _find(db, user, symbol.upper())
    if item is None:
        raise HTTPException(status_code=404, detail="Not in watchlist")
    db.delete(item)
    db.commit()
