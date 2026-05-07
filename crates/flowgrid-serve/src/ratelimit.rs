use std::sync::{Arc, Mutex};

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Clone, Debug)]
pub struct RateLimitState {
    pub rps: u64,
    pub capacity: u64,
    slot: Arc<Mutex<(f64, u64)>>, // tokens (fractional), last_seen_ms
}

impl RateLimitState {
    /// Token bucket whose capacity defaults to **`rps`** (≈ “one refill per second at steady state”).
    pub fn new(rps: u64) -> Self {
        let r = rps.max(1);
        Self::with_capacity(r, r)
    }

    /// Burst capacity **`capacity`**, refill rate **`rps`** tokens per wall-clock second.
    pub fn with_capacity(rps: u64, capacity: u64) -> Self {
        let r = rps.max(1);
        let cap_u64 = capacity.max(1);
        let cap = cap_u64 as f64;
        let now = now_ms();
        Self {
            rps: r,
            capacity: cap_u64,
            slot: Arc::new(Mutex::new((cap, now))),
        }
    }

    pub fn allow(&self) -> bool {
        let now = now_ms();
        let mut guard = match self.slot.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let (tokens, last) = *guard;
        let elapsed_ms = now.saturating_sub(last) as f64;
        let mut t =
            (tokens + elapsed_ms * (self.rps as f64 / 1000.0)).min(self.capacity as f64);
        if t >= 1.0 {
            t -= 1.0;
            *guard = (t, now);
            true
        } else {
            *guard = (t, now);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn burst_ceiling_respects_capacity() {
        let rps = 1_000_000_u64;
        let cap = 5_u64;
        let rl = RateLimitState::with_capacity(rps, cap);
        let mut ok = 0u32;
        for _ in 0..cap * 2 {
            if rl.allow() {
                ok += 1;
            }
        }
        assert_eq!(ok as u64, cap, "immediate burst should saturate at capacity");
    }

    #[test]
    fn steady_rps_keeps_most_requests_when_spaced() {
        let rl = RateLimitState::new(10);
        thread::sleep(Duration::from_millis(210));
        assert!(rl.allow());
        thread::sleep(Duration::from_millis(110));
        assert!(rl.allow());
    }
}
