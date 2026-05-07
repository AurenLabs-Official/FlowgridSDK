use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct RateLimitState {
    pub rps: u64,
    pub slot: Arc<Mutex<(u64, u64)>>,
}

impl RateLimitState {
    pub fn new(rps: u64) -> Self {
        Self {
            rps: rps.max(1),
            slot: Arc::new(Mutex::new((0, 0))),
        }
    }

    pub fn allow(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut guard = self.slot.lock().expect("rate limit lock");
        if guard.0 != now {
            guard.0 = now;
            guard.1 = 0;
        }
        if guard.1 >= self.rps {
            return false;
        }
        guard.1 += 1;
        true
    }
}
