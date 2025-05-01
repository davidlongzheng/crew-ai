#pragma once

#include <vector>
#include <algorithm>
#include "types.h"

// Split a hand of cards by suit
inline std::vector<std::vector<Card>> split_by_suit(const std::vector<Card> &hand)
{
    // Sort the hand by suit and rank
    std::vector<Card> sorted_hand = hand;
    std::sort(sorted_hand.begin(), sorted_hand.end(),
              [](const Card &a, const Card &b)
              {
                  if (a.suit != b.suit)
                  {
                      return a.suit < b.suit;
                  }
                  return a.rank < b.rank;
              });

    std::vector<std::vector<Card>> result;
    if (sorted_hand.empty())
    {
        return result;
    }

    // Group cards by suit
    std::vector<Card> current_suit;
    int prev_suit = sorted_hand[0].suit;

    for (const auto &card : sorted_hand)
    {
        if (card.suit != prev_suit)
        {
            result.push_back(current_suit);
            current_suit.clear();
            prev_suit = card.suit;
        }
        current_suit.push_back(card);
    }

    // Add the last suit group
    if (!current_suit.empty())
    {
        result.push_back(current_suit);
    }

    return result;
}
