// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
use std::fmt::Debug;

use lance_core::Result;

use crate::scalar::inverted::scorer::Scorer;

use super::{PostingIterator, Wand, WandDocuments};

impl<'a, S: Scorer, D: WandDocuments> Wand<'a, S, D> {
    pub(super) fn check_positions(&self, slop: i32) -> Result<bool> {
        #[cfg(test)]
        {
            self.phrase_position_checks
                .set(self.phrase_position_checks.get() + 1);
        }
        if slop == 0 {
            return self.check_exact_positions();
        }

        let mut position_iters = self
            .current_doc_postings()
            .into_iter()
            .map(PostingIterator::position_cursor)
            .collect::<Result<Vec<_>>>()?;
        position_iters.sort_unstable_by_key(|iter| iter.position_in_query);

        loop {
            let mut max_relative_pos = None;
            let mut all_same = true;
            for window in position_iters.windows(2) {
                let last = window[0].relative_position();
                let next = window[1].relative_position();
                let (Some(last), Some(next)) = (last, next) else {
                    return Ok(false);
                };

                let move_to = if last > next {
                    last
                } else {
                    std::cmp::max(last + 1, next - slop)
                };
                max_relative_pos = max_relative_pos.max(Some(move_to));
                if !(last <= next && next <= last + slop) {
                    all_same = false;
                    break;
                }
            }

            if all_same {
                return Ok(true);
            }

            position_iters.iter_mut().for_each(|iter| {
                iter.advance_to_relative(max_relative_pos.unwrap());
            });
        }
    }

    /// Exact-phrase check for the bulk conjunction path, where every clause
    /// is a parked `lead` iterator. Semantically identical to
    /// [`Self::check_exact_positions`].
    pub(super) fn check_exact_positions_bulk(&self) -> Result<bool> {
        #[cfg(test)]
        {
            self.phrase_position_checks
                .set(self.phrase_position_checks.get() + 1);
        }
        exact_phrase_positions_match(
            self.lead.len(),
            |index| {
                self.lead[index]
                    .doc()
                    .map(|doc| doc.frequency())
                    .unwrap_or(u32::MAX)
            },
            |index| self.lead[index].position_cursor(),
        )
    }

    pub(super) fn check_exact_positions(&self) -> Result<bool> {
        let postings = self.current_doc_postings();
        exact_phrase_positions_match(
            postings.len(),
            |index| {
                postings[index]
                    .doc()
                    .map(|doc| doc.frequency())
                    .unwrap_or(u32::MAX)
            },
            |index| postings[index].position_cursor(),
        )
    }
}

/// Decode exact-phrase position lists in current-doc frequency order.
///
/// Term frequency equals the position count, so a stopword on this document
/// is decoded last. If the two rarest lists share no phrase base, the rest
/// (typically `"the"` / `"of"`) are never unpacked.
fn exact_phrase_positions_match<'a>(
    num_clauses: usize,
    frequency: impl Fn(usize) -> u32,
    decode: impl FnMut(usize) -> Result<PositionCursor<'a>>,
) -> Result<bool> {
    const MAX_INLINE_CLAUSES: usize = 16;
    if num_clauses == 0 {
        return Ok(false);
    }
    if num_clauses > MAX_INLINE_CLAUSES {
        let mut order: Vec<usize> = (0..num_clauses).collect();
        order.sort_unstable_by_key(|&index| frequency(index));
        let mut cursors: Vec<Option<PositionCursor<'a>>> = (0..num_clauses).map(|_| None).collect();
        return exact_phrase_scan(&order, &mut cursors, decode);
    }
    let mut order = [0usize; MAX_INLINE_CLAUSES];
    for (index, slot) in order.iter_mut().enumerate().take(num_clauses) {
        *slot = index;
    }
    order[..num_clauses].sort_unstable_by_key(|&index| frequency(index));
    let mut cursors: [Option<PositionCursor<'a>>; MAX_INLINE_CLAUSES] =
        std::array::from_fn(|_| None);
    exact_phrase_scan(&order[..num_clauses], &mut cursors, decode)
}

fn exact_phrase_scan<'a>(
    order: &[usize],
    cursors: &mut [Option<PositionCursor<'a>>],
    mut decode: impl FnMut(usize) -> Result<PositionCursor<'a>>,
) -> Result<bool> {
    let num_clauses = order.len();
    if num_clauses == 0 {
        return Ok(false);
    }
    let rarest = order[0];
    cursors[rarest] = Some(decode(rarest)?);
    let (n_pos, anchor_offset) = {
        let cursor = cursors[rarest]
            .as_ref()
            .expect("rarest clause cursor was just populated");
        (cursor.len(), cursor.position_in_query as u32)
    };
    if num_clauses == 1 {
        return Ok(n_pos > 0);
    }

    // Every clause's target is `anchor_position - anchor_offset + query_offset`,
    // which is strictly increasing in the anchor. Two consequences the loop
    // below exploits:
    //
    //   * a follower is never searched from the start. Each cursor keeps its own
    //     index and only moves forward, so a failed probe leaves it sitting on
    //     the first position above the target - which is also the smallest
    //     position any *later* anchor can align with.
    //   * the anchor can therefore be jumped straight to that position instead
    //     of stepping over every candidate in between. None of them can match:
    //     their targets all land in the gap the follower was just shown to have.
    //
    // Both moves start from the current index and only move forward, so the
    // per-document cost is bounded by the decoded position counts rather than
    // by their product. The jump is taken only when it skips at least one
    // anchor position, which needs no counter, budget or mode switch: the two
    // regimes (far-apart lists vs interleaved lists) separate themselves.
    let mut anchor_idx = 0usize;
    'anchor: while anchor_idx < n_pos {
        #[cfg(test)]
        {
            count_anchor_steps();
        }
        let anchor_position = cursors[rarest]
            .as_ref()
            .expect("rarest clause cursor was just populated")
            .positions
            .as_slice()[anchor_idx];
        let Some(base) = anchor_position.checked_sub(anchor_offset) else {
            anchor_idx += 1;
            continue 'anchor;
        };
        for &index in &order[1..num_clauses] {
            if cursors[index].is_none() {
                cursors[index] = Some(decode(index)?);
            }
            let query_offset = cursors[index]
                .as_ref()
                .expect("phrase clause cursor was just populated")
                .position_in_query as u32;
            let Some(target) = base.checked_add(query_offset) else {
                return Ok(false);
            };
            let Some(position) = cursors[index]
                .as_mut()
                .expect("phrase clause cursor was just populated")
                .advance_to_at_least(target)
            else {
                // No follower position reaches `target`, and every later anchor
                // only raises it, so no remaining base can align either.
                return Ok(false);
            };
            if position == target {
                continue;
            }
            let next_anchor = position
                .saturating_sub(query_offset)
                .saturating_add(anchor_offset);
            anchor_idx = {
                let anchor_positions = cursors[rarest]
                    .as_ref()
                    .expect("rarest clause cursor was just populated")
                    .positions
                    .as_slice();
                // Only pay for the jump when it actually skips something. If the
                // very next anchor position is already below the bound, the jump
                // would land one step ahead anyway, and a plain step gets there
                // without the extra binary search. This is what separates the
                // two regimes: on disjoint lists the first jump crosses the whole
                // anchor, while on interleaved lists every bound lands one
                // position ahead and the scan degrades to stepping.
                match anchor_positions.get(anchor_idx + 1) {
                    Some(&next_position) if next_position < next_anchor => {
                        anchor_idx
                            + anchor_positions[anchor_idx..].partition_point(|&p| p < next_anchor)
                    }
                    _ => anchor_idx + 1,
                }
            };
            continue 'anchor;
        }
        return Ok(true);
    }
    Ok(false)
}

#[cfg(test)]
thread_local! {
    pub(super) static POSITION_CURSOR_CALLS: Cell<usize> = const { Cell::new(0) };
    static PHRASE_ANCHOR_STEPS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
thread_local! {
    // Test-only: counting anchor steps is off unless a test arms it, because the
    // scan loop runs it once per anchor position. Rustdoc does not document
    // macro invocations, so this is a plain comment.
    static COUNT_ANCHOR_STEPS: Cell<bool> = const { Cell::new(false) };
}

/// Armed per test thread: a process-wide flag would let one parallel test that
/// finishes early switch counting off for another still running.
#[cfg(test)]
fn count_anchor_steps() {
    if COUNT_ANCHOR_STEPS.with(|armed| armed.get()) {
        PHRASE_ANCHOR_STEPS.with(|steps| steps.set(steps.get() + 1));
    }
}

#[cfg(test)]
pub(super) fn take_position_cursor_calls() -> usize {
    POSITION_CURSOR_CALLS.with(|calls| calls.replace(0))
}

/// Anchor positions visited by the last `exact_phrase_scan`. Test-only: it is
/// the only way to assert that the scan skips without a timing threshold.
#[cfg(test)]
fn take_anchor_steps() -> usize {
    PHRASE_ANCHOR_STEPS.with(|steps| steps.replace(0))
}

/// Run an exact-phrase check with anchor-step counting armed.
#[cfg(test)]
pub(super) fn counted_anchor_steps<F: FnOnce() -> bool>(check: F) -> (bool, usize) {
    COUNT_ANCHOR_STEPS.with(|armed| armed.set(true));
    let _ = take_anchor_steps();
    let matched = check();
    let steps = take_anchor_steps();
    COUNT_ANCHOR_STEPS.with(|armed| armed.set(false));
    (matched, steps)
}

#[derive(Debug)]
pub(super) struct RecycledPositionValues<'a> {
    values: Option<Vec<u32>>,
    pool: &'a RefCell<Option<Vec<u32>>>,
}

impl<'a> RecycledPositionValues<'a> {
    pub(super) fn new(values: Vec<u32>, pool: &'a RefCell<Option<Vec<u32>>>) -> Self {
        Self {
            values: Some(values),
            pool,
        }
    }

    fn as_slice(&self) -> &[u32] {
        self.values
            .as_deref()
            .expect("position values are present until drop")
    }
}

impl Drop for RecycledPositionValues<'_> {
    fn drop(&mut self) {
        let values = self
            .values
            .take()
            .expect("position values are present until drop");
        let mut pool = self.pool.borrow_mut();
        if pool.is_none() {
            *pool = Some(values);
        }
    }
}

#[derive(Debug)]
pub(super) enum PositionValues<'a> {
    Recycled(RecycledPositionValues<'a>),
    Owned(Vec<u32>),
}

impl<'a> PositionValues<'a> {
    pub(super) fn as_slice(&self) -> &[u32] {
        match self {
            Self::Recycled(values) => values.as_slice(),
            Self::Owned(values) => values.as_slice(),
        }
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

#[derive(Debug)]
pub(super) struct PositionCursor<'a> {
    pub(super) positions: PositionValues<'a>,
    pub position_in_query: i32,
    index: usize,
}

impl<'a> PositionCursor<'a> {
    pub(super) fn new(positions: PositionValues<'a>, position_in_query: i32) -> Self {
        Self {
            positions,
            position_in_query,
            index: 0,
        }
    }

    fn len(&self) -> usize {
        self.positions.len()
    }

    fn relative_position(&self) -> Option<i32> {
        self.positions
            .as_slice()
            .get(self.index)
            .map(|position| *position as i32 - self.position_in_query)
    }

    /// Advance to the first position at or above `target` and return it, or
    /// `None` once the cursor is exhausted.
    ///
    /// `target` must be non-decreasing across calls: the search is confined to
    /// the not-yet-consumed suffix, so the cursor never rewinds and a returned
    /// position is never below `target`. When the cursor already sits past
    /// `target` the search is skipped entirely.
    fn advance_to_at_least(&mut self, target: u32) -> Option<u32> {
        if self.index >= self.len() {
            return None;
        }
        let values = self.positions.as_slice();
        // The cursor has reached `target`, or is already past it. Every earlier
        // position is below the previous target and hence below this one, so the
        // insertion point has not moved: no search is needed, and the current
        // position is either the match or the bound the caller wants.
        if values[self.index] >= target {
            return Some(values[self.index]);
        }
        // Targets only grow, so the next position is a frequent answer. Check it
        // before paying for a search: a suffix binary search costs a dependent
        // load per level and its landmarks move as `index` advances, which is
        // what makes interleaved lists slow. On a mismatch here the cursor ends
        // up one position short of the target on every step.
        if let Some(next) = values
            .get(self.index + 1)
            .copied()
            .filter(|&next| next >= target)
        {
            self.index += 1;
            return Some(next);
        }
        self.index += values[self.index..].partition_point(|&pos| pos < target);
        values.get(self.index).copied()
    }

    fn advance_to_relative(&mut self, least_relative_pos: i32) {
        if self.index >= self.len() {
            return;
        }
        let least_pos = least_relative_pos + self.position_in_query;
        let least_pos = least_pos.max(0) as u32;
        let values = self.positions.as_slice();
        self.index += values[self.index..].partition_point(|&pos| pos < least_pos);
    }
}
