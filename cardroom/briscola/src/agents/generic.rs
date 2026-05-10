use crate::env::env::BriscolaEnv;
use crate::env::player::BriscolaAction;                                                                         
                                      
pub trait BriscolaAgent {
    fn select_action(&self, env: &BriscolaEnv) -> BriscolaAction;
    fn name(&self) -> &str;
}