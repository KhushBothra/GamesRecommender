"""Retrieve the closest games for a weighted descriptor query."""

from Backend.tasks.recommendation.recommend import RecommendationError, recommend_games

__all__ = ["RecommendationError", "recommend_games"]
