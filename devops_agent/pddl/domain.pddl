(define (domain devops)
  (:requirements :typing :numeric-fluents :action-costs)

  (:types
    service configopt - object
  )

  (:predicates
    (running ?s - service)
    (mem_ok ?s - service)
    (config_applied ?s - service ?c - configopt)
  )

  (:functions
    (min_safe_mem ?s - service) - number
    (total-cost) - number
  )

  (:action set_mem
    :parameters (?s - service ?c - configopt)
    :precondition (and
      (not (mem_ok ?s))
      (not (config_applied ?s ?c))
    )
    :effect (and
      (mem_ok ?s)
      (config_applied ?s ?c)
      (increase (total-cost) (min_safe_mem ?s))
    )
  )

  (:action deploy
    :parameters (?s - service)
    :precondition (and
      (not (running ?s))
      (mem_ok ?s)
    )
    :effect (and
      (running ?s)
      (increase (total-cost) 1)
    )
  )
)
