(define (domain devops)
  (:requirements :strips :typing :negative-preconditions)

  (:types service configopt - object)

  (:predicates
    (running    ?s - service)
    (mem_ok     ?s - service)
    (config_set ?s - service ?o - configopt)
    (config_ok  ?s - service ?o - configopt))

  (:action set_mem
    :parameters (?s - service)
    :precondition (not (mem_ok ?s))
    :effect (mem_ok ?s))

  (:action set_config
    :parameters (?s - service ?o - configopt)
    :precondition (not (config_set ?s ?o))
    :effect (config_set ?s ?o))

  (:action deploy
    :parameters (?s - service ?o - configopt)
    :precondition (and (not (running ?s))
                       (mem_ok ?s)
                       (config_set ?s ?o)
                       (config_ok ?s ?o))
    :effect (running ?s))
)
