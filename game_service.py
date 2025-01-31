from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from openai import OpenAI
import os
from dotenv import load_dotenv
import uvicorn
import json
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="AI Game Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
)

games: Dict[str, Dict] = {}

class GameMove(BaseModel):
    game_id: Optional[str]
    move: str
    game_type: str = "tic-tac-toe"

class GameResponse(BaseModel):
    game_id: str
    ai_move: str
    board_state: List[str]
    game_status: str
    message: str

def create_game_prompt(game_type: str, board_state: List[str], player_move: str) -> str:
    if game_type == "tic-tac-toe":
        board_display = "\n".join([
            f"{board_state[i]} {board_state[i+1]} {board_state[i+2]}"
            for i in range(0, 9, 3)
        ])
        return f"""
        You are playing Tic-tac-toe against a human. You are 'O' and the human is 'X'.
        Current board state (positions 0-8):
        {board_display}
        
        Human just played 'X' at position {player_move}.
        
        Choose a valid empty position (marked with '-') for your move.
        Respond with ONLY a JSON object in this EXACT format, where move MUST be a single digit 0-8:
        {{"move": 4, "reasoning": "why you chose this move", "game_status": "ongoing"}}
        
        IMPORTANT: 
        - 'move' must be a single number (0-8)
        - Only choose positions marked with '-'
        - game_status must be: "ongoing", "won", or "draw"
        """
    return ""

def initialize_board(game_type: str) -> List[str]:
    if game_type == "tic-tac-toe":
        return ["-" for _ in range(9)]
    return []

def check_valid_move(board: List[str], move: int) -> bool:
    if not (0 <= move <= 8):
        return False
    return board[move] == "-"

def check_winner(board: List[str]) -> str:
    win_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    
    for combo in win_combinations:
        if board[combo[0]] != "-" and board[combo[0]] == board[combo[1]] == board[combo[2]]:
            return board[combo[0]]
    
    if "-" not in board:
        return "draw"
    
    return "ongoing"

@app.post("/game/move", response_model=GameResponse)
async def make_move(move_data: GameMove):
    try:
        if not move_data.game_id:
            game_id = str(uuid.uuid4())
            games[game_id] = {
                "board": initialize_board(move_data.game_type),
                "game_type": move_data.game_type
            }
        else:
            game_id = move_data.game_id
            if game_id not in games:
                raise HTTPException(status_code=404, detail="Game not found")

        game = games[game_id]
        board = game["board"]

        try:
            player_move = int(move_data.move)
            if not check_valid_move(board, player_move):
                raise HTTPException(status_code=400, detail="Invalid move")
            board[player_move] = "X"
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid move format")

        game_state = check_winner(board)
        if game_state in ["X", "draw"]:
            return GameResponse(
                game_id=game_id,
                ai_move="-1",
                board_state=board,
                game_status="won" if game_state == "X" else "draw",
                message="You won!" if game_state == "X" else "It's a draw!"
            )

        prompt = create_game_prompt(move_data.game_type, board, move_data.move)
        logger.info(f"Sending prompt to AI: {prompt}")
        
        completion = client.chat.completions.create(
            model="deepseek/deepseek-r1-distill-qwen-1.5b",
            messages=[
                {"role": "system", "content": "You are playing a game of Tic-tac-toe. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ]
        )

        try:
            response_content = completion.choices[0].message.content
            logger.info(f"AI response: {response_content}")
            ai_response = json.loads(response_content)
            
            move_str = str(ai_response["move"]).strip().split('-')[0]  # Handle cases like "2-"
            ai_move = int(move_str)
            
            if not (0 <= ai_move <= 8):
                raise ValueError(f"AI move {ai_move} is out of range")
            
            if not check_valid_move(board, ai_move):
                raise HTTPException(status_code=500, detail=f"AI made invalid move: {ai_move}")
            
            board[ai_move] = "O"
            
            game_state = check_winner(board)
            if game_state == "O":
                ai_response["game_status"] = "won"
                ai_response["reasoning"] = "I won!"
            elif game_state == "draw":
                ai_response["game_status"] = "draw"
                ai_response["reasoning"] = "It's a draw!"
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Error parsing AI response: {e}")
            empty_positions = [i for i, val in enumerate(board) if val == "-"]
            if empty_positions:
                ai_move = empty_positions[0]
                board[ai_move] = "O"
                ai_response = {
                    "game_status": "ongoing",
                    "reasoning": "Fallback move"
                }
            else:
                raise HTTPException(status_code=500, detail="No valid moves available")

        games[game_id]["board"] = board

        return GameResponse(
            game_id=game_id,
            ai_move=str(ai_move),
            board_state=board,
            game_status=ai_response.get("game_status", "ongoing"),
            message=ai_response.get("reasoning", "Move completed")
        )

    except Exception as e:
        logger.error(f"Error in make_move: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/game/{game_id}")
async def get_game_state(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    return games[game_id]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 