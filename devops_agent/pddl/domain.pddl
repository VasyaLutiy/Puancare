(define (domain devops)
  (:requirements :typing :numeric-fluents :action-costs)

  (:types
    service configopt - object
  )

  (:predicates
    (running ?s - service)
    (config_applied ?s - service ?c - configopt)
  )

  (:functions
    (mem ?s - service)
    (min_safe_mem ?s - service)
    (total-cost) - number
  )

  (:action set_mem
    :parameters (?s - service ?c - configopt)
    :precondition (and
      (not (config_applied ?s ?c))
    )
    :effect (and
      (config_applied ?s ?c)
      (assign (mem ?s) (min_safe_mem ?s))
      (increase (total-cost) (min_safe_mem ?s))
    )
  )

  (:action deploy
    :parameters (?s - service)
    :precondition (and
      (not (running ?s))
      (>= (mem ?s) (min_safe_mem ?s))
    )
    :effect (and
      (running ?s)
      (increase (total-cost) 1)
    )
  )
)
