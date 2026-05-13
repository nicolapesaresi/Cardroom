use crate::env::env::BriscolaObs;
use crate::env::player::BriscolaAction;                                                                         
                                      
pub trait BriscolaAgent {
    fn select_action(&self, obs: BriscolaObs) -> BriscolaAction;
    fn name(&self) -> &str;
}